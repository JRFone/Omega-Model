#define MyAppName "Omega FISH Model"
#define MyAppVersion "1.4.1"
#define MyAppPublisher "Omega FISH Model Project"
#define MyAppExeName "Omega FISH Model.exe"

[Setup]
AppId={{7A91D20D-7C8A-4C65-A5E4-CAD6D28F4A73}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={localappdata}\Programs\Omega FISH Model
DefaultGroupName=Omega FISH Model
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=..\release
OutputBaseFilename=Omega_FISH_Model_Setup_{#MyAppVersion}
SetupIconFile=..\assets\omega_fish.ico
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayIcon={app}\{#MyAppExeName}

[Files]
Source: "..\dist\Omega FISH Model\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\Omega FISH Model"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\Omega FISH Model"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional icons:"; Flags: unchecked

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch Omega FISH Model"; Flags: nowait postinstall skipifsilent
