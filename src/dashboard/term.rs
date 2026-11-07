//! The terminal side: raw mode and the alternate screen while the dashboard runs, restored on
//! every way out (return, error, panic: [`Session`] restores on drop and a panic hook restores
//! before the panic message prints), and the draw / key loop ([`run`]). [`Feed`] carries what
//! the client's network tasks receive to the loop.

use std::io::{self, Stdout};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, Once};
use std::time::{Duration, Instant};

use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::crossterm::event::{self, Event, KeyEventKind};
use ratatui::crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::crossterm::{cursor, execute};

use super::logs::LogLine;
use super::snapshot::Snapshot;
use super::ui::{self, Action, App, Conn};

/// How often the screen redraws (and keys are read).
const FRAME: Duration = Duration::from_millis(100);

/// What the network side has received since the loop last looked.
#[derive(Debug, Default)]
pub struct Inbox {
    pub snaps: Vec<Snapshot>,
    pub logs: Vec<LogLine>,
    /// The log source starts over (reconnected): drop the lines shown so far.
    pub logs_reset: bool,
    pub conn: Option<Conn>,
}

/// The network tasks' side of [`Inbox`].
#[derive(Debug, Default)]
pub struct Feed {
    inbox: Mutex<Inbox>,
}

impl Feed {
    fn with(&self, f: impl FnOnce(&mut Inbox)) {
        let mut g = self.inbox.lock().unwrap_or_else(|e| e.into_inner());
        f(&mut g);
    }

    pub fn snapshot(&self, s: Snapshot) {
        // The loop only needs the newest few; a stalled screen must not pile them up.
        self.with(|i| {
            if i.snaps.len() >= 8 {
                i.snaps.remove(0);
            }
            i.snaps.push(s);
        });
    }

    pub fn logs(&self, mut lines: Vec<LogLine>, reset: bool) {
        self.with(|i| {
            if reset {
                i.logs.clear();
                i.logs_reset = true;
            }
            i.logs.append(&mut lines);
            let over = i.logs.len().saturating_sub(super::logs::CAPACITY);
            i.logs.drain(..over);
        });
    }

    pub fn conn(&self, c: Conn) {
        self.with(|i| i.conn = Some(c));
    }

    pub fn take(&self) -> Inbox {
        let mut g = self.inbox.lock().unwrap_or_else(|e| e.into_inner());
        std::mem::take(&mut *g)
    }
}

impl App {
    /// Applies what the feed received.
    pub fn ingest(&mut self, inbox: Inbox) {
        if let Some(c) = inbox.conn {
            self.conn = c;
        }
        for s in inbox.snaps {
            self.push_snapshot(s);
        }
        if inbox.logs_reset || !inbox.logs.is_empty() {
            self.push_logs(inbox.logs, inbox.logs_reset);
        }
    }
}

/// Puts the terminal back: cooked mode, the main screen, the cursor shown. Safe to call
/// more than once.
pub fn restore() {
    let _ = disable_raw_mode();
    let _ = execute!(io::stdout(), LeaveAlternateScreen, cursor::Show);
    super::logs::set_muted(false);
}

/// The terminal while the dashboard owns it; dropping it restores the terminal.
pub struct Session {
    pub terminal: Terminal<CrosstermBackend<Stdout>>,
}

impl Session {
    /// Raw mode, the alternate screen, logs held back. Installs the panic hook once.
    pub fn enter() -> io::Result<Self> {
        static HOOK: Once = Once::new();
        HOOK.call_once(|| {
            let prev = std::panic::take_hook();
            std::panic::set_hook(Box::new(move |info| {
                restore();
                prev(info);
            }));
        });
        super::logs::set_muted(true);
        if let Err(e) = enable_raw_mode() {
            super::logs::set_muted(false);
            return Err(e);
        }
        let mut out = io::stdout();
        if let Err(e) = execute!(out, EnterAlternateScreen, cursor::Hide) {
            restore();
            return Err(e);
        }
        match Terminal::new(CrosstermBackend::new(out)) {
            Ok(terminal) => Ok(Self { terminal }),
            Err(e) => {
                restore();
                Err(e)
            }
        }
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        restore();
    }
}

/// Why [`run`] returned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Exit {
    /// `q`.
    Quit,
    /// Ctrl-C (raw mode turns it into a key).
    Interrupt,
    /// [`request_stop`]: the process is about to exit.
    Stopped,
}

static STOP: AtomicBool = AtomicBool::new(false);

/// Asks [`run`] to return before its next frame, e.g. when the server stops on its own (bind
/// error, SIGTERM): the terminal must be back to normal before the process exits.
pub fn request_stop() {
    STOP.store(true, Ordering::Relaxed);
}

/// Draws `app` and reads keys until `q` or Ctrl-C. `poll` feeds the app before each frame
/// (new snapshots and log lines). The terminal is restored when this returns, also on error.
pub fn run(app: &mut App, mut poll: impl FnMut(&mut App)) -> io::Result<Exit> {
    let mut session = Session::enter()?;
    session.terminal.clear()?;
    loop {
        if STOP.load(Ordering::Relaxed) {
            return Ok(Exit::Stopped);
        }
        let started = Instant::now();
        poll(app);
        session.terminal.draw(|f| ui::draw(f, app))?;
        let wait = FRAME.saturating_sub(started.elapsed());
        if event::poll(wait)? {
            // Drain what is queued so typing never lags a frame per key.
            loop {
                if let Event::Key(k) = event::read()?
                    && k.kind != KeyEventKind::Release
                {
                    match app.handle_key(k) {
                        Action::None => {}
                        Action::Quit => return Ok(Exit::Quit),
                        Action::Interrupt => return Ok(Exit::Interrupt),
                    }
                }
                if !event::poll(Duration::ZERO)? {
                    break;
                }
            }
        }
    }
}
