<?php
header('Content-Type: application/json; charset=utf-8');

if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    echo json_encode(["status" => "error", "message" => "Only POST requests are permitted."]);
    exit;
}

if (!isset($_FILES['flower_image']) || $_FILES['flower_image']['error'] !== UPLOAD_ERR_OK) {
    echo json_encode(["status" => "error", "message" => "No valid photo was uploaded."]);
    exit;
}

$uploadDir = __DIR__ . '/uploads/';
if (!is_dir($uploadDir)) {
    mkdir($uploadDir, 0755, true);
}

$fileTmpPath = $_FILES['flower_image']['tmp_name'];
$fileName = 'cnn_img_' . time() . '_' . uniqid() . '.' . pathinfo($_FILES['flower_image']['name'], PATHINFO_EXTENSION);
$destPath = $uploadDir . $fileName;

if (move_uploaded_file($fileTmpPath, $destPath)) {
    
    // Command string execution setup
    $pythonExecutable = 'python3'; // Replace with virtualenv path if needed (e.g. /var/www/env/bin/python)
    $scriptPath = __DIR__ . '/classify.py';
    
    $command = escapeshellcmd("$pythonExecutable " . escapeshellarg($scriptPath) . " " . escapeshellarg($destPath));
    
    // Execute Python CNN process
    $output = shell_exec($command);

    // Clean up temporary image file
    if (file_exists($destPath)) {
        unlink($destPath);
    }

    if ($output) {
        echo $output;
    } else {
        echo json_encode(["status" => "error", "message" => "CNN Python script execution returned empty output."]);
    }
} else {
    echo json_encode(["status" => "error", "message" => "Failed to save upload image to server."]);
}
?>