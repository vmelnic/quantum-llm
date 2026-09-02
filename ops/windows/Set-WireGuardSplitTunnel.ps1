param(
    [string]$TunnelName = "vmelnic",
    [string]$AllowedIPs = "10.10.88.0/24",
    [switch]$CheckOnly
)

. (Join-Path $PSScriptRoot "Common.ps1")
Initialize-ExperimentDirectories

Add-Type -TypeDefinition @'
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;

public static class WireGuardDpapi
{
    [StructLayout(LayoutKind.Sequential)]
    private struct DataBlob
    {
        public int Length;
        public IntPtr Data;
    }

    [DllImport("crypt32.dll", SetLastError = true, CharSet = CharSet.Unicode)]
    private static extern bool CryptProtectData(
        ref DataBlob input,
        string description,
        IntPtr optionalEntropy,
        IntPtr reserved,
        IntPtr prompt,
        uint flags,
        ref DataBlob output);

    [DllImport("crypt32.dll", SetLastError = true, CharSet = CharSet.Unicode)]
    private static extern bool CryptUnprotectData(
        ref DataBlob input,
        out IntPtr description,
        IntPtr optionalEntropy,
        IntPtr reserved,
        IntPtr prompt,
        uint flags,
        ref DataBlob output);

    [DllImport("kernel32.dll")]
    private static extern IntPtr LocalFree(IntPtr memory);

    private const uint UiForbidden = 0x1;

    private static DataBlob Allocate(byte[] value)
    {
        DataBlob blob = new DataBlob();
        blob.Length = value.Length;
        blob.Data = Marshal.AllocHGlobal(value.Length);
        Marshal.Copy(value, 0, blob.Data, value.Length);
        return blob;
    }

    private static void ClearAndFreeInput(DataBlob blob)
    {
        if (blob.Data == IntPtr.Zero)
            return;
        for (int index = 0; index < blob.Length; ++index)
            Marshal.WriteByte(blob.Data, index, 0);
        Marshal.FreeHGlobal(blob.Data);
    }

    private static byte[] CopyAndFreeOutput(DataBlob blob)
    {
        if (blob.Data == IntPtr.Zero)
            return new byte[0];
        byte[] value = new byte[blob.Length];
        Marshal.Copy(blob.Data, value, 0, blob.Length);
        for (int index = 0; index < blob.Length; ++index)
            Marshal.WriteByte(blob.Data, index, 0);
        LocalFree(blob.Data);
        return value;
    }

    public static byte[] Unprotect(byte[] cipher, out string description)
    {
        DataBlob input = Allocate(cipher);
        DataBlob output = new DataBlob();
        IntPtr descriptionPointer = IntPtr.Zero;
        try
        {
            if (!CryptUnprotectData(ref input, out descriptionPointer,
                    IntPtr.Zero, IntPtr.Zero, IntPtr.Zero, UiForbidden,
                    ref output))
                throw new Win32Exception(Marshal.GetLastWin32Error());
            description = descriptionPointer == IntPtr.Zero
                ? null
                : Marshal.PtrToStringUni(descriptionPointer);
            return CopyAndFreeOutput(output);
        }
        finally
        {
            ClearAndFreeInput(input);
            if (descriptionPointer != IntPtr.Zero)
                LocalFree(descriptionPointer);
        }
    }

    public static byte[] Protect(byte[] plain, string description)
    {
        DataBlob input = Allocate(plain);
        DataBlob output = new DataBlob();
        try
        {
            if (!CryptProtectData(ref input, description, IntPtr.Zero,
                    IntPtr.Zero, IntPtr.Zero, UiForbidden, ref output))
                throw new Win32Exception(Marshal.GetLastWin32Error());
            return CopyAndFreeOutput(output);
        }
        finally
        {
            ClearAndFreeInput(input);
        }
    }
}
'@

if ($TunnelName -notmatch '^[A-Za-z0-9._-]+$') {
    throw "TunnelName contains unsupported characters"
}
if ($AllowedIPs -notmatch '^10\.10\.88\.0/24$') {
    throw "This operation is restricted to the validated WireGuard subnet"
}
$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
if ($identity.User.Value -ne 'S-1-5-18') {
    throw "WireGuard DPAPI configuration must be updated as Local System"
}

$configurationPath = Join-Path `
    (Join-Path $env:ProgramFiles "WireGuard\Data\Configurations") `
    "$TunnelName.conf.dpapi"
$wireGuard = Join-Path $env:ProgramFiles "WireGuard\wg.exe"
$serviceName = 'WireGuardTunnel$' + $TunnelName
$statusName = "wireguard-split-tunnel.json"
$backupDirectory = Join-Path $script:RepoRoot "work\backups\wireguard"

if (-not (Test-Path -LiteralPath $configurationPath -PathType Leaf)) {
    throw "WireGuard configuration is missing: $configurationPath"
}
if (-not (Test-Path -LiteralPath $wireGuard -PathType Leaf)) {
    throw "WireGuard CLI is missing: $wireGuard"
}

$cipher = [IO.File]::ReadAllBytes($configurationPath)
$plain = $null
$updatedPlain = $null
$configurationWritten = $false
$backupPath = $null
try {
    $storedName = $null
    $plain = [WireGuardDpapi]::Unprotect($cipher, [ref]$storedName)
    if ($storedName -ne $TunnelName) {
        throw "DPAPI configuration name does not match the tunnel name"
    }
    $encoding = [Text.UTF8Encoding]::new($false)
    $text = $encoding.GetString($plain)
    $allowedMatches = [regex]::Matches(
        $text, '(?im)^\s*AllowedIPs\s*=\s*(.+?)\s*$')
    if ($allowedMatches.Count -ne 1) {
        throw "Expected exactly one AllowedIPs declaration"
    }
    if (-not [regex]::IsMatch($text, '(?im)^\s*PrivateKey\s*=')) {
        throw "WireGuard configuration has no interface private key"
    }
    if (-not [regex]::IsMatch($text, '(?im)^\s*Address\s*=')) {
        throw "WireGuard configuration has no interface address"
    }
    $originalAllowedIPs = [string]$allowedMatches[0].Groups[1].Value

    if ($CheckOnly) {
        $status = [PSCustomObject]@{
            status = "ready"
            tunnel = $TunnelName
            original_allowed_ips = $originalAllowedIPs
            requested_allowed_ips = $AllowedIPs
            configuration_path = $configurationPath
            running_as_system = $true
        }
        Write-JsonArtifact -Value $status -Name $statusName | Out-Null
        exit 0
    }

    $matchedLine = [string]$allowedMatches[0].Value
    $updatedLine = [regex]::Replace(
        $matchedLine,
        '(?i)(AllowedIPs\s*=\s*).+$',
        ('${1}' + $AllowedIPs))
    $updatedText = $text.Remove(
        $allowedMatches[0].Index,
        $allowedMatches[0].Length).Insert(
            $allowedMatches[0].Index,
            $updatedLine)
    $updatedPlain = $encoding.GetBytes($updatedText)
    $updatedCipher = [WireGuardDpapi]::Protect($updatedPlain, $TunnelName)

    New-Item -ItemType Directory -Path $backupDirectory -Force | Out-Null
    $backupPath = Join-Path $backupDirectory `
        ("$TunnelName.conf.dpapi." + [DateTime]::UtcNow.ToString(
            "yyyyMMddTHHmmssZ") + ".bak")
    [IO.File]::WriteAllBytes($backupPath, $cipher)

    $temporaryPath = "$configurationPath.tmp"
    [IO.File]::WriteAllBytes($temporaryPath, $updatedCipher)
    Move-Item -LiteralPath $temporaryPath -Destination $configurationPath -Force
    $configurationWritten = $true

    foreach ($prefix in @("0.0.0.0/1", "128.0.0.0/1")) {
        Get-NetRoute -DestinationPrefix $prefix -PolicyStore ActiveStore `
                -ErrorAction SilentlyContinue |
            Remove-NetRoute -Confirm:$false -ErrorAction SilentlyContinue
    }
    Get-NetRoute -DestinationPrefix "10.10.88.0/24" `
            -PolicyStore ActiveStore -ErrorAction SilentlyContinue |
        Remove-NetRoute -Confirm:$false -ErrorAction SilentlyContinue

    Restart-Service -Name $serviceName -Force -ErrorAction Stop
    (Get-Service -Name $serviceName).WaitForStatus(
        [System.ServiceProcess.ServiceControllerStatus]::Running,
        [TimeSpan]::FromSeconds(30))
    Start-Sleep -Seconds 2

    $runtimeAllowed = @(& $wireGuard show $TunnelName allowed-ips 2>$null |
        ForEach-Object { [string]$_ })
    if ($runtimeAllowed.Count -ne 1 -or
        $runtimeAllowed[0] -notmatch '\t10\.10\.88\.0/24$') {
        throw "WireGuard did not load the split-tunnel AllowedIPs"
    }
    $sshRoute = Find-NetRoute -RemoteIPAddress "10.10.88.2"
    if (@($sshRoute.InterfaceAlias) -notcontains $TunnelName) {
        throw "The management address is not routed through WireGuard"
    }
    $publicRoute = Find-NetRoute -RemoteIPAddress "1.1.1.1"
    if (@($publicRoute.InterfaceAlias) -contains $TunnelName) {
        throw "Public traffic is still routed through WireGuard"
    }
    $https = Test-NetConnection -ComputerName "huggingface.co" -Port 443 `
        -InformationLevel Detailed -WarningAction SilentlyContinue
    if (-not $https.TcpTestSucceeded) {
        throw "Direct Ethernet HTTPS validation failed"
    }

    $status = [PSCustomObject]@{
        status = "configured"
        tunnel = $TunnelName
        original_allowed_ips = $originalAllowedIPs
        active_allowed_ips = $AllowedIPs
        management_interface = [string]@($sshRoute.InterfaceAlias)[0]
        public_interface = [string]@($publicRoute.InterfaceAlias)[0]
        https_succeeded = $true
        backup_path = $backupPath
        configured_utc = [DateTime]::UtcNow.ToString("o")
    }
    Write-JsonArtifact -Value $status -Name $statusName | Out-Null
}
catch {
    $failure = $_.Exception.Message
    if ($configurationWritten -and $null -ne $backupPath -and
        (Test-Path -LiteralPath $backupPath -PathType Leaf)) {
        Copy-Item -LiteralPath $backupPath -Destination $configurationPath -Force
        try {
            Restart-Service -Name $serviceName -Force -ErrorAction Stop
        }
        catch {}
    }
    $status = [PSCustomObject]@{
        status = "failed"
        tunnel = $TunnelName
        error = $failure
        rolled_back = $configurationWritten
        backup_path = $backupPath
        failed_utc = [DateTime]::UtcNow.ToString("o")
    }
    Write-JsonArtifact -Value $status -Name $statusName | Out-Null
    throw
}
finally {
    if ($null -ne $plain) { [Array]::Clear($plain, 0, $plain.Length) }
    if ($null -ne $updatedPlain) {
        [Array]::Clear($updatedPlain, 0, $updatedPlain.Length)
    }
}
