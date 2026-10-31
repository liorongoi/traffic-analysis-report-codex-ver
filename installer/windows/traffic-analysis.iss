; Build-time only: compile with Inno Setup on Windows after staging/verification.
#define AppVersion "0.1.0"
[Setup]
AppId=RoadTrafficAnalysisIndependent
AppName=RoadLoom Preview
AppVersion={#AppVersion}
DefaultDirName={localappdata}\Programs\RoadTrafficAnalysis
DefaultGroupName=RoadLoom
PrivilegesRequired=lowest
DisableProgramGroupPage=yes
OutputDir=..\..\dist\windows
OutputBaseFilename=RoadLoom-Preview-Setup
SetupIconFile=..\..\web\roadloom.ico
UninstallDisplayIcon={app}\web\roadloom.ico
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
CloseApplications=yes
UninstallDisplayName=RoadLoom Preview
[InstallDelete]
; Remove only OCR model files shipped by superseded plate-recognition builds.
; Current plate handling keeps crop images and review history, but has no OCR.
Type: files; Name: "{app}\model-packs\license-plate-yolox\plate_ocr_arabic_dict.json"
Type: files; Name: "{app}\model-packs\license-plate-yolox\plate_ocr_arabic_ppocrv3_mobile.onnx"
Type: files; Name: "{app}\model-packs\license-plate-yolox\plate_ocr_global_cct_s_v2.onnx"
Type: files; Name: "{app}\model-packs\license-plate-yolox\plate_ocr_global_cct_s_v2_model_config.yaml"
Type: files; Name: "{app}\model-packs\license-plate-yolox\plate_ocr_global_cct_s_v2_plate_config.yaml"
[Files]
Source: "..\..\staging\windows\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
[Icons]
Name: "{group}\RoadLoom"; Filename: "{app}\scripts\Start-Traffic-Analysis.cmd"; WorkingDir: "{app}"; IconFilename: "{app}\web\roadloom.ico"
Name: "{group}\Uninstall"; Filename: "{uninstallexe}"
[Run]
Filename: "{app}\runtime\python.exe"; Parameters: "-I ""{app}\scripts\verify-install.py"" ""{app}"""; StatusMsg: "Verifying bundled offline files..."; Flags: waituntilterminated runhidden; AfterInstall: CheckVerification
[Code]
procedure CheckVerification;
begin
  if not FileExists(ExpandConstant('{app}\verified-receipt.json')) then
    RaiseException('Offline verification did not complete. See verification-error.txt in the application folder. No downloads were attempted.');
end;
