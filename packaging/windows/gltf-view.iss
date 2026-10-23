; Inno Setup script for gltf-view. Built by .github/workflows/release.yml:
;   iscc /DAppVersion=X.Y.Z /DSourceDir=<dir with gltf-view.exe> packaging\windows\gltf-view.iss
; Installs per-user by default (no admin prompt); "Install for all users" is offered too.

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef SourceDir
  #define SourceDir "..\..\target\release"
#endif

[Setup]
AppId={{6F1D2C3B-8E4A-4F5B-9C7D-2A1B3C4D5E6F}
AppName=gltf-view
AppVersion={#AppVersion}
AppVerName=gltf-view {#AppVersion}
AppPublisher=Jason Beverage
AppPublisherURL=https://github.com/jasonbeverage/gltf-view
AppSupportURL=https://github.com/jasonbeverage/gltf-view/issues
AppUpdatesURL=https://github.com/jasonbeverage/gltf-view/releases
VersionInfoVersion={#AppVersion}
DefaultDirName={autopf}\gltf-view
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
LicenseFile=..\..\LICENSE
SetupIconFile=..\..\assets\icon.ico
UninstallDisplayIcon={app}\gltf-view.exe
OutputDir=..\..\dist
OutputBaseFilename=gltf-view-windows-x64-setup
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ChangesEnvironment=yes
ChangesAssociations=yes
CloseApplications=yes

[Tasks]
Name: "addtopath"; Description: "Add gltf-view to the PATH (run ""gltf-view model.glb"" from a terminal)"
Name: "associate"; Description: "Open .gltf and .glb files with gltf-view"
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "{#SourceDir}\gltf-view.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\..\README.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\..\LICENSE"; DestDir: "{app}"; DestName: "LICENSE.txt"; Flags: ignoreversion

[Icons]
Name: "{autoprograms}\gltf-view"; Filename: "{app}\gltf-view.exe"
Name: "{autodesktop}\gltf-view"; Filename: "{app}\gltf-view.exe"; Tasks: desktopicon

[Run]
; The app registers itself for the user who ran the installer (not the elevated admin).
Filename: "{app}\gltf-view.exe"; Parameters: "--register"; Flags: runasoriginaluser waituntilterminated; Tasks: associate
Filename: "{app}\gltf-view.exe"; Description: "{cm:LaunchProgram,gltf-view}"; Flags: nowait postinstall skipifsilent runasoriginaluser

[UninstallRun]
Filename: "{app}\gltf-view.exe"; Parameters: "--unregister"; Flags: runhidden waituntilterminated; RunOnceId: "Unassociate"

[Code]
const
  UserEnvKey = 'Environment';
  MachineEnvKey = 'SYSTEM\CurrentControlSet\Control\Session Manager\Environment';

function EnvRoot: Integer;
begin
  if IsAdminInstallMode then Result := HKEY_LOCAL_MACHINE else Result := HKEY_CURRENT_USER;
end;

function EnvKey: String;
begin
  if IsAdminInstallMode then Result := MachineEnvKey else Result := UserEnvKey;
end;

procedure AddToPath(Dir: String);
var
  Paths: String;
begin
  if not RegQueryStringValue(EnvRoot, EnvKey, 'Path', Paths) then Paths := '';
  if Pos(';' + Uppercase(Dir) + ';', ';' + Uppercase(Paths) + ';') > 0 then exit;
  if (Paths <> '') and (Paths[Length(Paths)] <> ';') then Paths := Paths + ';';
  RegWriteExpandStringValue(EnvRoot, EnvKey, 'Path', Paths + Dir);
end;

procedure RemoveFromPath(Dir: String);
var
  Paths, Padded: String;
  P: Integer;
begin
  if not RegQueryStringValue(EnvRoot, EnvKey, 'Path', Paths) then exit;
  Padded := ';' + Paths + ';';
  P := Pos(';' + Uppercase(Dir) + ';', Uppercase(Padded));
  if P = 0 then exit;
  Delete(Padded, P, Length(Dir) + 1);
  // Strip the padding added above.
  if (Length(Padded) > 0) and (Padded[1] = ';') then Delete(Padded, 1, 1);
  if (Length(Padded) > 0) and (Padded[Length(Padded)] = ';') then Delete(Padded, Length(Padded), 1);
  RegWriteExpandStringValue(EnvRoot, EnvKey, 'Path', Padded);
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if (CurStep = ssPostInstall) and WizardIsTaskSelected('addtopath') then
    AddToPath(ExpandConstant('{app}'));
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usPostUninstall then
    RemoveFromPath(ExpandConstant('{app}'));
end;
