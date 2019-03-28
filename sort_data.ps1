$dir = "./data"

$items = Get-ChildItem $dir -File
$imgdir = (mkdir $dir"/img" -Force).FullName
$annDir = (mkdir $dir"/annotations" -Force).FullName
foreach($item in $items){
    if($item.Name.endswith(".jpg")){
        Move-Item -Path $item.FullName -Destination $imgdir -Force 
    }
    elseif($item.Name.endswith(".xml")){
        Move-Item -Path $item.FullName -Destination $annDir -Force 
    }
    else{
        Remove-Item -Path $item.FullName -Force
    }
}
#   Where BaseName -match '(\d{2})_(\d{2})_\d{4}_DSC_\d{4}'|
#     Group {$Matches[1]+'-'+$Matches[2]}|
#       ForEach{MD $_.Name;$_.Group|Move -Dest $_.Name}