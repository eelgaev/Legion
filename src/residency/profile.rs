//! Expert profiles: `(layer, expert)` pairs ranked by how often the router picks them, which
//! decide the VRAM expert cache's first contents and its per-layer sizes.
//!
//! The file format is Strata's `profile.bin` version 1 (MIT, eelgaev/Strata-AC922
//! `include/strata/core/expert_cache.hpp` and `tools/make_profile.py`), so a profile written by
//! either engine seeds the other: the 4 bytes `STRP`, five little-endian u32 (version 1,
//! n_layers, n_expert, slots, n_ranked), `n_ranked` pairs of u16 (layer, expert), then an
//! `n_layers × n_expert` i32 table of each pair's rank (-1: unranked). Readers use the pairs.

use std::io::{self, Read, Write};
use std::path::Path;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Profile {
    pub n_layers: usize,
    pub n_expert: usize,
    /// Hottest first.
    pub ranked: Vec<(usize, u32)>,
}

fn invalid(msg: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg)
}

impl Profile {
    /// Ranks every `(layer, expert)` pair with nonzero heat, hottest first (ties by layer,
    /// then expert). `heat[layer][expert]`.
    pub fn from_heat(heat: &[Vec<f32>]) -> Self {
        let n_expert = heat.iter().map(Vec::len).max().unwrap_or(0);
        let mut pairs: Vec<(f32, usize, u32)> = heat
            .iter()
            .enumerate()
            .flat_map(|(l, h)| {
                h.iter()
                    .enumerate()
                    .filter(|(_, x)| **x > 0.0)
                    .map(move |(e, &x)| (x, l, e as u32))
            })
            .collect();
        pairs.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
        Self {
            n_layers: heat.len(),
            n_expert,
            ranked: pairs.into_iter().map(|(_, l, e)| (l, e)).collect(),
        }
    }

    /// Per-layer heat that reproduces this ranking (the top pair gets the most), for seeding
    /// a cache's statistics: `heat[layer][expert]`, scaled to at most `top`.
    pub fn heat(&self, top: f32) -> Vec<Vec<f32>> {
        let mut h = vec![vec![0.0; self.n_expert]; self.n_layers];
        let n = self.ranked.len().max(1) as f32;
        for (r, &(l, e)) in self.ranked.iter().enumerate() {
            h[l][e as usize] = top * (n - r as f32) / n;
        }
        h
    }

    pub fn read(path: &Path) -> io::Result<Self> {
        let mut bytes = Vec::new();
        std::fs::File::open(path)?.read_to_end(&mut bytes)?;
        Self::parse(&bytes).map_err(|e| invalid(format!("{}: {e}", path.display())))
    }

    pub fn parse(b: &[u8]) -> Result<Self, String> {
        if b.len() < 24 || &b[..4] != b"STRP" {
            return Err("not an expert profile (no STRP header)".into());
        }
        let u = |i: usize| u32::from_le_bytes(b[4 + 4 * i..8 + 4 * i].try_into().unwrap());
        let (version, n_layers, n_expert, _slots, n_ranked) = (u(0), u(1), u(2), u(3), u(4));
        if version != 1 {
            return Err(format!("profile version {version}, this reads version 1"));
        }
        let n_ranked = n_ranked as usize;
        let pairs = b
            .get(24..24 + 4 * n_ranked)
            .ok_or("the ranked list is truncated")?;
        let mut ranked = Vec::with_capacity(n_ranked);
        for p in pairs.as_chunks::<4>().0 {
            let l = u16::from_le_bytes([p[0], p[1]]) as usize;
            let e = u16::from_le_bytes([p[2], p[3]]) as u32;
            if l >= n_layers as usize || e >= n_expert {
                return Err(format!("pair (layer {l}, expert {e}) out of range"));
            }
            ranked.push((l, e));
        }
        Ok(Self {
            n_layers: n_layers as usize,
            n_expert: n_expert as usize,
            ranked,
        })
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(24 + 4 * self.ranked.len());
        out.extend_from_slice(b"STRP");
        let n = self.ranked.len() as u32;
        for v in [1, self.n_layers as u32, self.n_expert as u32, n, n] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        let mut table = vec![-1i32; self.n_layers * self.n_expert];
        for (r, &(l, e)) in self.ranked.iter().enumerate() {
            out.extend_from_slice(&(l as u16).to_le_bytes());
            out.extend_from_slice(&(e as u16).to_le_bytes());
            table[l * self.n_expert + e as usize] = r as i32;
        }
        for v in table {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    /// Writes atomically (a `.tmp` file renamed over `path`).
    pub fn write(&self, path: &Path) -> io::Result<()> {
        let tmp = path.with_extension("tmp");
        {
            let mut f = std::fs::File::create(&tmp)?;
            f.write_all(&self.to_bytes())?;
            f.sync_all()?;
        }
        std::fs::rename(&tmp, path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heat_ranks_round_trip_through_the_file_format() {
        let heat = vec![vec![0.0, 5.0, 1.0], vec![7.0, 0.0, 5.0]];
        let p = Profile::from_heat(&heat);
        assert_eq!(p.ranked, vec![(1, 0), (0, 1), (1, 2), (0, 2)]);
        let bytes = p.to_bytes();
        assert_eq!(bytes.len(), 24 + 4 * 4 + 4 * 6);
        assert_eq!(Profile::parse(&bytes).unwrap(), p);
        let h = p.heat(4.0);
        assert!(h[1][0] > h[0][1] && h[0][1] > h[1][2] && h[1][2] > h[0][2] && h[0][2] > 0.0);
        assert_eq!(h[0][0], 0.0);
    }

    #[test]
    fn bad_files_are_refused() {
        assert!(Profile::parse(b"NOPE").is_err());
        let mut b = Profile::from_heat(&[vec![1.0]]).to_bytes();
        b[4] = 2;
        assert!(Profile::parse(&b).unwrap_err().contains("version 2"));
        let mut b = Profile::from_heat(&[vec![1.0]]).to_bytes();
        b[24] = 9; // layer 9 of 1
        assert!(Profile::parse(&b).unwrap_err().contains("out of range"));
        let b = Profile::from_heat(&[vec![1.0]]).to_bytes();
        assert!(Profile::parse(&b[..25]).is_err());
    }
}
