use clap::Args;
use legion::hw::HardwareInfo;
use legion::residency::MemoryModel;

#[derive(Args)]
pub struct InfoArgs {
    /// Machine-readable output.
    #[arg(long)]
    json: bool,
    /// Also measure how fast each GPU reads host memory on its NUMA node (in place and by
    /// DMA, for the expert residency layer). Runs a short kernel on every GPU.
    #[arg(long)]
    probe: bool,
}

pub fn run(args: InfoArgs) -> anyhow::Result<()> {
    let info = HardwareInfo::detect();
    let memory = MemoryModel::detect(&info, args.probe);
    if args.json {
        let mut v = serde_json::to_value(&info)?;
        v["residency"] = serde_json::to_value(&memory)?;
        println!("{}", serde_json::to_string_pretty(&v)?);
    } else {
        print!("{}", info.render());
        print!("{}", memory.render());
    }
    Ok(())
}
