// Embeds the icon and version info into the Windows executable.
fn main() {
    println!("cargo:rerun-if-changed=assets/icon.ico");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        let mut res = winresource::WindowsResource::new();
        res.set_icon("assets/icon.ico");
        res.set("FileDescription", "gltf-view - glTF/GLB viewer");
        res.set("ProductName", "gltf-view");
        res.compile().expect("failed to embed Windows resources");
    }
}
