//! Per-user file association for .gltf / .glb (Windows only, no admin rights needed).

#[cfg(windows)]
mod imp {
    use anyhow::Result;
    use winreg::RegKey;
    use winreg::enums::{HKEY_CURRENT_USER, KEY_READ};

    const PROG_ID: &str = "gltf-view.model";
    const APP: &str = "gltf-view";
    const EXTS: [&str; 2] = [".gltf", ".glb"];

    fn notify_shell() {
        use windows_sys::Win32::UI::Shell::{SHCNE_ASSOCCHANGED, SHCNF_IDLIST, SHChangeNotify};
        unsafe { SHChangeNotify(SHCNE_ASSOCCHANGED as i32, SHCNF_IDLIST, std::ptr::null(), std::ptr::null()) };
    }

    pub fn register() -> Result<()> {
        let exe = std::env::current_exe()?;
        let exe = exe.to_string_lossy();
        let hkcu = RegKey::predef(HKEY_CURRENT_USER);

        let (prog, _) = hkcu.create_subkey(format!(r"Software\Classes\{PROG_ID}"))?;
        prog.set_value("", &"glTF 3D Model")?;
        prog.create_subkey("DefaultIcon")?.0.set_value("", &format!("\"{exe}\",0"))?;
        prog.create_subkey(r"shell\open\command")?.0.set_value("", &format!("\"{exe}\" \"%1\""))?;

        let (app, _) = hkcu.create_subkey(r"Software\Classes\Applications\gltf-view.exe")?;
        app.set_value("FriendlyAppName", &APP)?;
        app.create_subkey(r"shell\open\command")?.0.set_value("", &format!("\"{exe}\" \"%1\""))?;
        let (types, _) = app.create_subkey("SupportedTypes")?;

        let (caps, _) = hkcu.create_subkey(format!(r"Software\{APP}\Capabilities"))?;
        caps.set_value("ApplicationName", &APP)?;
        caps.set_value("ApplicationDescription", &"Fast glTF / GLB viewer")?;
        let (fa, _) = caps.create_subkey("FileAssociations")?;

        for ext in EXTS {
            let (k, _) = hkcu.create_subkey(format!(r"Software\Classes\{ext}"))?;
            k.set_value("", &PROG_ID)?;
            k.set_value("Content Type", &if ext == ".glb" { "model/gltf-binary" } else { "model/gltf+json" })?;
            k.create_subkey("OpenWithProgids")?.0.set_value(PROG_ID, &"")?;
            types.set_value(ext, &"")?;
            fa.set_value(ext, &PROG_ID)?;
        }
        hkcu.create_subkey(r"Software\RegisteredApplications")?
            .0
            .set_value(APP, &format!(r"Software\{APP}\Capabilities"))?;
        notify_shell();

        // Windows keeps an explicit per-user choice (UserChoice) that apps cannot
        // override; in that case open the Default Apps page for gltf-view.
        if !is_registered() {
            let _ = std::process::Command::new("explorer.exe")
                .arg(format!("ms-settings:defaultapps?registeredAppUser={APP}"))
                .spawn();
        }
        Ok(())
    }

    pub fn unregister() -> Result<()> {
        let hkcu = RegKey::predef(HKEY_CURRENT_USER);
        let _ = hkcu.delete_subkey_all(format!(r"Software\Classes\{PROG_ID}"));
        let _ = hkcu.delete_subkey_all(r"Software\Classes\Applications\gltf-view.exe");
        let _ = hkcu.delete_subkey_all(format!(r"Software\{APP}"));
        if let Ok(k) = hkcu.open_subkey_with_flags(r"Software\RegisteredApplications", winreg::enums::KEY_WRITE) {
            let _ = k.delete_value(APP);
        }
        for ext in EXTS {
            if let Ok(k) = hkcu.open_subkey_with_flags(format!(r"Software\Classes\{ext}"), winreg::enums::KEY_ALL_ACCESS) {
                if k.get_value::<String, _>("").ok().as_deref() == Some(PROG_ID) {
                    let _ = k.delete_value("");
                }
                if let Ok(o) = k.open_subkey_with_flags("OpenWithProgids", winreg::enums::KEY_WRITE) {
                    let _ = o.delete_value(PROG_ID);
                }
            }
        }
        notify_shell();
        Ok(())
    }

    /// True when double-clicking both extensions opens gltf-view.
    pub fn is_registered() -> bool {
        let hkcu = RegKey::predef(HKEY_CURRENT_USER);
        let Ok(cmd) = hkcu.open_subkey_with_flags(format!(r"Software\Classes\{PROG_ID}\shell\open\command"), KEY_READ) else {
            return false;
        };
        let exe = std::env::current_exe().map(|e| e.to_string_lossy().into_owned()).unwrap_or_default();
        if !cmd.get_value::<String, _>("").unwrap_or_default().contains(&exe) {
            return false;
        }
        EXTS.iter().all(|ext| {
            let choice = hkcu
                .open_subkey(format!(r"Software\Microsoft\Windows\CurrentVersion\Explorer\FileExts\{ext}\UserChoice"))
                .and_then(|k| k.get_value::<String, _>("ProgId"));
            match choice {
                Ok(p) => p == PROG_ID || p.eq_ignore_ascii_case("Applications\\gltf-view.exe"),
                Err(_) => hkcu
                    .open_subkey(format!(r"Software\Classes\{ext}"))
                    .and_then(|k| k.get_value::<String, _>(""))
                    .is_ok_and(|p| p == PROG_ID),
            }
        })
    }
}

#[cfg(not(windows))]
mod imp {
    pub fn register() -> anyhow::Result<()> {
        anyhow::bail!("file association is managed by your desktop environment; install packaging/gltf-view.desktop")
    }
    pub fn unregister() -> anyhow::Result<()> {
        Ok(())
    }
    pub fn is_registered() -> bool {
        false
    }
}

pub use imp::{is_registered, register, unregister};

pub fn is_supported() -> bool {
    cfg!(windows)
}
