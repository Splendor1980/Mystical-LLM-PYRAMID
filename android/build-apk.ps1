# Gradle-less build of Mystica-debug.apk.
# Requires: Android SDK (platf. android-34, build-tools 34.0.0), JDK (see vars below).
param(
    [string]$Sdk  = "D:\AndroidSetup\sdk",
    [string]$Jdk  = "D:\AndroidSetup\jdk",
    [string]$Stag  = "debug",        # keystore alias tag (used for auto-made debug keystore)
    [string]$SrcDir = (Join-Path $PSScriptRoot "src"),
    [string]$OutDir = (Join-Path $PSScriptRoot "..\build\android"),
    [string]$Keystore = "",          # release keystore path (non-empty => release signing)
    [string]$StorePass = "",
    [string]$AliasName = ""
)

# aapt2 cannot handle non-ASCII (e.g. Cyrillic) paths -> stage sources under
# an ASCII work dir and copy the finished apk back.
$workRoot = Join-Path $env:TEMP "mystica-build"
$work = Join-Path $workRoot "apk"
$src  = Join-Path $work "src"
$out  = Join-Path $work "out"
if (Test-Path $work) { Remove-Item -Recurse -Force $work }
New-Item -ItemType Directory -Force -Path $work, $src, $out | Out-Null
Copy-Item -Recurse -Force (Join-Path $SrcDir "main") (Join-Path $src "main")
Copy-Item -Recurse -Force (Join-Path $SrcDir "core") (Join-Path $src "core")

$ErrorActionPreference = "Stop"

$bt  = Join-Path $Sdk "build-tools\34.0.0"
$apt = Join-Path $Sdk "platforms\android-34\android.jar"
$classes = Join-Path $out "classes"
$dexDir  = Join-Path $out "dex"
$resFlat = Join-Path $out "res"

$jbin = Join-Path $Jdk "bin"
$javac = Join-Path $jbin "javac.exe"
$java  = Join-Path $jbin "java.exe"
$jar   = Join-Path $jbin "jar.exe"
$keytool = Join-Path $jbin "keytool.exe"
$aapt2 = Join-Path $bt "aapt2.exe"
$d8Jar = Join-Path $bt "lib\d8.jar"
$signJar = Join-Path $bt "lib\apksigner.jar"
$zipalign = Join-Path $bt "zipalign.exe"

New-Item -ItemType Directory -Force -Path $classes, $dexDir, $resFlat | Out-Null

Write-Host "=== 1/7 resources (aapt2 compile) ==="
& $aapt2 compile --dir (Join-Path $src "main\res") -o $resFlat
if ($LASTEXITCODE -ne 0) { throw "aapt2 compile failed" }

Write-Host "=== 2/7 resources (aapt2 link) ==="
$flatFiles = (Get-ChildItem $resFlat -Filter *.flat -Recurse | ForEach-Object { $_.FullName })
$baseApk = Join-Path $out "base.apk"
& $aapt2 link -o $baseApk -I $apt `
    --manifest (Join-Path $src "main\AndroidManifest.xml") `
    -A (Join-Path $src "main\assets") `
    --auto-add-overlay `
    --java (Join-Path $out "gen") `
    $flatFiles
if ($LASTEXITCODE -ne 0) { throw "aapt2 link failed" }

Write-Host "=== 3/7 java (javac vs android.jar) ==="
$genR = Get-ChildItem (Join-Path $out "gen") -Filter R.java -Recurse | ForEach-Object { $_.FullName }
$allJava = @()
$allJava += $genR
$allJava += Get-ChildItem (Join-Path $src "core\java") -Filter *.java -Recurse | ForEach-Object { $_.FullName }
$allJava += Get-ChildItem (Join-Path $src "main\java") -Filter *.java -Recurse | ForEach-Object { $_.FullName }
& $javac -encoding UTF-8 -classpath $apt -d $classes $allJava
if ($LASTEXITCODE -ne 0) { throw "javac failed" }

Write-Host "=== 4/7 dex (d8) ==="
$classesJar = Join-Path $out "classes.jar"
& $jar cf $classesJar -C $classes .
if ($LASTEXITCODE -ne 0) { throw "jar failed" }
& $java -cp $d8Jar com.android.tools.r8.D8 --release --lib $apt --min-api 26 --output $dexDir $classesJar
if ($LASTEXITCODE -ne 0) { throw "d8 failed" }

Write-Host "=== 5/7 inject classes.dex into apk ==="
& $jar uf $baseApk -C $dexDir classes.dex
if ($LASTEXITCODE -ne 0) { throw "jar update failed" }

Write-Host "=== 6/7 zipalign ==="
$aligned = Join-Path $out "aligned.apk"
& $zipalign -f 4 $baseApk $aligned
if ($LASTEXITCODE -ne 0) { throw "zipalign failed" }

Write-Host "=== 7/7 sign ==="
if ($Keystore -ne "" -and $StorePass -ne "" -and $AliasName -ne "") {
    $ks      = $Keystore
    $ksPass  = $StorePass
    $alias   = $AliasName
    $apkBase = "Mystica-release.apk"
} else {
    $ks      = Join-Path $workRoot "mystica-debug.keystore"
    $ksPass  = "android"
    $alias   = $Stag
    $apkBase = "Mystica-debug.apk"
    if (-not (Test-Path $ks)) {
        & $keytool -genkeypair -v -keystore $ks -storepass $ksPass `
            -alias $alias -keypass $ksPass -keyalg RSA -keysize 2048 -validity 10000 `
            -dname "CN=Android Debug,O=Mystica,C=RU"
        if ($LASTEXITCODE -ne 0) { throw "keytool failed" }
    }
}
$finalApk = Join-Path $out $apkBase
& $java -cp $signJar com.android.apksigner.ApkSignerTool sign --ks $ks --ks-pass pass:$ksPass --ks-key-alias $alias --out $finalApk $aligned
if ($LASTEXITCODE -ne 0) { throw "apksigner failed" }

& $java -cp $signJar com.android.apksigner.ApkSignerTool verify $finalApk
if ($LASTEXITCODE -ne 0) { throw "apksigner verify failed" }

# copy finished apk back to the project build dir (non-ASCII-safe)
if (Test-Path $OutDir) { Remove-Item -Recurse -Force $OutDir }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
Copy-Item -Force $finalApk $OutDir
$final = Join-Path $OutDir $apkBase

$size = (Get-Item $final).Length
Write-Host ""
Write-Host "DONE: $final"
Write-Host ("size: {0:N0} bytes" -f $size)