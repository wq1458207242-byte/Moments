Param(
  [string]$RepoName = "Moments2",
  [ValidateSet("private","public")][string]$Visibility = "private",
  [string]$RemoteUrl,
  [switch]$UseExtraHeader
)

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
  Write-Error "未检测到 git，请先在系统安装 Git。"
  exit 1
}

$token = $env:GITHUB_TOKEN

if (-not $RemoteUrl) {
  if (-not $token) {
    Write-Error "未检测到环境变量 GITHUB_TOKEN。请临时设置：`$env:GITHUB_TOKEN = '你的令牌'`"
    exit 1
  }
  $headers = @{
    Authorization = "Bearer $token"
    "User-Agent" = "TraeUpload"
    Accept = "application/vnd.github+json"
  }

  try {
    $user = Invoke-RestMethod -Method GET -Uri "https://api.github.com/user" -Headers $headers
  } catch {
    Write-Error "获取 GitHub 用户信息失败：$($_.Exception.Message)"
    exit 1
  }
  $login = $user.login

  $body = @{ name = $RepoName; private = ($Visibility -eq "private") } | ConvertTo-Json
  try {
    $repo = Invoke-RestMethod -Method POST -Uri "https://api.github.com/user/repos" -Headers $headers -Body $body
  } catch {
    try {
      $repo = Invoke-RestMethod -Method GET -Uri "https://api.github.com/repos/$login/$RepoName" -Headers $headers
    } catch {
      Write-Error "创建或获取仓库失败：$($_.Exception.Message)"
      exit 1
    }
  }
  $cloneUrl = $repo.clone_url
} else {
  $cloneUrl = $RemoteUrl
  $login = $null
}

Set-Location -Path "g:\Moments2"

if (-not (Test-Path ".git")) {
  git init
  git branch -M main
}

$uname = git config user.name
$uemail = git config user.email
if (-not $uname) { git config user.name "$login" }
if (-not $uemail) { git config user.email "$login@users.noreply.github.com" }

git add .
git commit -m "Initial import" --allow-empty

if ($RemoteUrl -and $UseExtraHeader -and $token) {
  $owner = $null
  if ($cloneUrl -match 'github\.com/([^/]+)/') {
    $owner = $Matches[1]
  }
  if (-not $owner) { $owner = "x-access-token" }
  $bytes = [Text.Encoding]::ASCII.GetBytes("${owner}:${token}")
  $basic = [Convert]::ToBase64String($bytes)
  git -c ("http.extraheader=Authorization: Basic {0}" -f $basic) push -u $cloneUrl main
} else {
  $remotes = git remote
  # 如果提供令牌，使用 x-access-token 嵌入到远程URL以避免交互式认证
  if ($token -and $cloneUrl -match '^https://github\.com/.+\.git$') {
    $secureUrl = $cloneUrl -replace '^https://', ("https://x-access-token:{0}@" -f $token)
  } else {
    $secureUrl = $cloneUrl
  }
  if ($remotes -notcontains "origin") {
    git remote add origin $secureUrl
  } else {
    git remote set-url origin $secureUrl
  }
  git push -u origin main
}

if ($login) {
  Write-Host ("已推送到：https://github.com/{0}/{1}" -f $login, $RepoName)
} else {
  Write-Host ("已推送到：{0}" -f $cloneUrl)
}
