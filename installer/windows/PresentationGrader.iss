#define MyAppName "Presentation Grader"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "Presentation Grader"
#ifndef SourceDir
  #error SourceDir must point to the prepared package directory
#endif

[Setup]
AppId={{E9B6F31D-2E44-4A90-93B0-8F32C8A1A741}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={localappdata}\Programs\Presentation Grader
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=..\..\dist
OutputBaseFilename=PresentationGraderSetup-Windows-x64
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64
SetupLogging=yes
CloseApplications=force
RestartApplications=no

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\Presentation Grader"; Filename: "{app}\Launch Presentation Grader.bat"; WorkingDir: "{app}"
Name: "{autodesktop}\Presentation Grader"; Filename: "{app}\Launch Presentation Grader.bat"; WorkingDir: "{app}"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"; Flags: unchecked

[Run]
Filename: "{app}\Launch Presentation Grader.bat"; Description: "Launch Presentation Grader"; WorkingDir: "{app}"; Flags: postinstall shellexec nowait skipifsilent
