"""
app.py

Flask application for Incipit Genie.

Transforms Word documents with endnotes into Word documents with incipit notes.
"""

import os
from flask import Flask, request, send_file, render_template_string, jsonify
from werkzeug.utils import secure_filename
from io import BytesIO

from document_processor import process_document
from link_activator import activate_links

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload

# HTML template embedded for simplicity
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Incipit Genie</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            min-height: 100vh;
            color: #e8e8e8;
            padding: 40px 20px;
        }
        
        .container {
            max-width: 800px;
            margin: 0 auto;
        }
        
        header {
            text-align: center;
            margin-bottom: 40px;
        }
        
        h1 {
            font-size: 3rem;
            background: linear-gradient(135deg, #e94560, #ff6b6b);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 10px;
        }
        
        .subtitle {
            font-size: 1.2rem;
            color: #a0a0a0;
        }
        
        .upload-section {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 20px;
            padding: 40px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            margin-bottom: 30px;
        }
        
        .drop-zone {
            border: 2px dashed rgba(233, 69, 96, 0.5);
            border-radius: 15px;
            padding: 60px 40px;
            text-align: center;
            transition: all 0.3s ease;
            cursor: pointer;
        }
        
        .drop-zone:hover, .drop-zone.dragover {
            border-color: #e94560;
            background: rgba(233, 69, 96, 0.1);
        }
        
        .drop-zone-icon {
            font-size: 4rem;
            margin-bottom: 20px;
        }
        
        .drop-zone-text {
            font-size: 1.2rem;
            margin-bottom: 10px;
        }
        
        .drop-zone-hint {
            color: #888;
            font-size: 0.9rem;
        }
        
        #fileInput {
            display: none;
        }
        
        .file-info {
            display: none;
            background: rgba(233, 69, 96, 0.1);
            border-radius: 10px;
            padding: 15px 20px;
            margin-top: 20px;
            align-items: center;
            justify-content: space-between;
        }
        
        .file-info.visible {
            display: flex;
        }
        
        .file-name {
            font-weight: 500;
        }
        
        .file-size {
            color: #888;
            font-size: 0.9rem;
        }
        
        .remove-file {
            background: none;
            border: none;
            color: #e94560;
            cursor: pointer;
            font-size: 1.5rem;
            padding: 5px;
        }
        
        .btn {
            display: inline-block;
            padding: 15px 40px;
            font-size: 1.1rem;
            font-weight: 600;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .btn-primary {
            background: linear-gradient(135deg, #e94560, #ff6b6b);
            color: white;
            width: 100%;
            margin-top: 20px;
        }
        
        .btn-primary:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 10px 30px rgba(233, 69, 96, 0.3);
        }
        
        .btn-primary:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        .progress-container {
            display: none;
            margin-top: 30px;
        }
        
        .progress-container.visible {
            display: block;
        }
        
        .progress-bar {
            height: 6px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 3px;
            overflow: hidden;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(135deg, #e94560, #ff6b6b);
            width: 0%;
            transition: width 0.3s ease;
        }
        
        .progress-text {
            text-align: center;
            margin-top: 10px;
            color: #888;
        }
        
        .result-section {
            display: none;
            text-align: center;
        }
        
        .result-section.visible {
            display: block;
        }
        
        .success-icon {
            font-size: 4rem;
            color: #4ade80;
            margin-bottom: 20px;
        }
        
        .error-icon {
            font-size: 4rem;
            color: #ef4444;
            margin-bottom: 20px;
        }
        
        .btn-download {
            background: linear-gradient(135deg, #4ade80, #22c55e);
            color: white;
        }
        
        .btn-download:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 30px rgba(74, 222, 128, 0.3);
        }
        
        .info-section {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 20px;
            padding: 30px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .info-section h2 {
            color: #e94560;
            margin-bottom: 20px;
            font-size: 1.3rem;
        }
        
        .info-section h3 {
            color: #ff6b6b;
            margin: 20px 0 10px;
            font-size: 1.1rem;
        }
        
        .info-section p, .info-section li {
            color: #a0a0a0;
            line-height: 1.6;
            margin-bottom: 10px;
        }
        
        .info-section ul {
            margin-left: 20px;
        }
        
        .example-box {
            background: rgba(0, 0, 0, 0.3);
            border-radius: 10px;
            padding: 20px;
            margin: 15px 0;
            font-family: monospace;
            font-size: 0.9rem;
        }
        
        .example-label {
            color: #e94560;
            font-weight: bold;
            margin-bottom: 5px;
        }
        
        .reset-btn {
            background: none;
            border: 1px solid rgba(255, 255, 255, 0.2);
            color: #a0a0a0;
            margin-top: 15px;
            padding: 10px 30px;
        }
        
        .reset-btn:hover {
            border-color: #e94560;
            color: #e94560;
        }
        
        .options-section {
            margin-top: 25px;
            padding: 20px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .options-row {
            display: flex;
            gap: 40px;
            flex-wrap: wrap;
        }
        
        .option-group {
            flex: 1;
            min-width: 200px;
        }
        
        .option-group label {
            display: block;
            margin-bottom: 10px;
            font-weight: 500;
            color: #e8e8e8;
        }
        
        .option-group input[type="number"] {
            width: 80px;
            padding: 10px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 6px;
            background: rgba(0, 0, 0, 0.3);
            color: #e8e8e8;
            font-size: 1rem;
        }
        
        .option-group input[type="number"]:focus {
            outline: none;
            border-color: #e94560;
        }
        
        .radio-group {
            display: flex;
            gap: 20px;
        }
        
        .radio-label {
            display: flex;
            align-items: center;
            cursor: pointer;
            padding: 8px 15px;
            border-radius: 6px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            transition: all 0.2s;
        }
        
        .radio-label:hover {
            border-color: #e94560;
        }
        
        .radio-label input[type="radio"] {
            margin-right: 8px;
            cursor: pointer;
        }
        
        .radio-label.selected {
            border-color: #e94560;
            background: rgba(233, 69, 96, 0.1);
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .processing {
            animation: pulse 1.5s infinite;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>✨ Incipit Genie</h1>
            <p class="subtitle">Transform endnotes into elegant incipit notes</p>
        </header>
        
        <div class="upload-section" id="uploadSection">
            <div class="drop-zone" id="dropZone">
                <div class="drop-zone-icon">📄</div>
                <div class="drop-zone-text">Drop your Word document here</div>
                <div class="drop-zone-hint">or click to browse (.docx files only)</div>
            </div>
            <input type="file" id="fileInput" accept=".docx">
            
            <div class="file-info" id="fileInfo">
                <div>
                    <div class="file-name" id="fileName"></div>
                    <div class="file-size" id="fileSize"></div>
                </div>
                <button class="remove-file" id="removeFile">&times;</button>
            </div>
            
            <div class="options-section">
                <div class="options-row">
                    <div class="option-group">
                        <label>🔢 Lead-in Text Length (words)</label>
                        <input type="number" id="wordCount" min="3" max="8" value="3">
                    </div>
                    <div class="option-group">
                        <label>📝 Lead-in Text Format</label>
                        <div class="radio-group">
                            <label class="radio-label selected">
                                <input type="radio" name="formatStyle" value="bold" checked> Bold
                            </label>
                            <label class="radio-label">
                                <input type="radio" name="formatStyle" value="italic"> Italic
                            </label>
                        </div>
                    </div>
                </div>
            </div>
            
            <button class="btn btn-primary" id="processBtn" disabled>
                Transform Document
            </button>
            
            <div class="progress-container" id="progressContainer">
                <div class="progress-bar">
                    <div class="progress-fill" id="progressFill"></div>
                </div>
                <div class="progress-text" id="progressText">Processing...</div>
            </div>
        </div>
        
        <div class="result-section" id="resultSection">
            <div id="resultContent"></div>
            <button class="btn reset-btn" id="resetBtn">Process Another Document</button>
        </div>
        
        <div class="info-section">
            <h2>How It Works</h2>
            <p>Incipit Genie transforms traditional endnotes into incipit (blind) notes by:</p>
            <ul>
                <li>Extracting meaningful phrases from your text using punctuation-based semantic analysis</li>
                <li>Removing superscript numbers from the main body</li>
                <li>Adding dynamic page references that update automatically</li>
            </ul>
            
            <h3>Example Transformation</h3>
            <div class="example-box">
                <div class="example-label">BEFORE:</div>
                <p>Main text: "Because it is the 'I' behind the 'eye' that does the seeing.<sup>1</sup>"</p>
                <p>Endnote: "1. Anaïs Nin, The Diary of Anaïs Nin..."</p>
            </div>
            <div class="example-box">
                <div class="example-label">AFTER:</div>
                <p>Main text: "Because it is the 'I' behind the 'eye' that does the seeing."</p>
                <p>Note: "<em>89</em>.&nbsp;&nbsp;<strong>Because it is:</strong> Anaïs Nin, The Diary of Anaïs Nin..."</p>
            </div>
            
            <h3>Updating Page Numbers</h3>
            <p>After opening the transformed document in Word, press <strong>Ctrl+A</strong> (select all) 
            then <strong>F9</strong> to update all page references.</p>
        </div>
    </div>
    
    <script>
        const dropZone = document.getElementById('dropZone');
        const fileInput = document.getElementById('fileInput');
        const fileInfo = document.getElementById('fileInfo');
        const fileName = document.getElementById('fileName');
        const fileSize = document.getElementById('fileSize');
        const removeFile = document.getElementById('removeFile');
        const processBtn = document.getElementById('processBtn');
        const progressContainer = document.getElementById('progressContainer');
        const progressFill = document.getElementById('progressFill');
        const progressText = document.getElementById('progressText');
        const uploadSection = document.getElementById('uploadSection');
        const resultSection = document.getElementById('resultSection');
        const resultContent = document.getElementById('resultContent');
        const resetBtn = document.getElementById('resetBtn');
        
        let selectedFile = null;
        
        // Drag and drop handlers
        dropZone.addEventListener('click', () => fileInput.click());
        
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });
        
        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('dragover');
        });
        
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleFile(files[0]);
            }
        });
        
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFile(e.target.files[0]);
            }
        });
        
        function handleFile(file) {
            if (!file.name.endsWith('.docx')) {
                alert('Please select a .docx file');
                return;
            }
            
            selectedFile = file;
            fileName.textContent = file.name;
            fileSize.textContent = formatFileSize(file.size);
            fileInfo.classList.add('visible');
            processBtn.disabled = false;
        }
        
        function formatFileSize(bytes) {
            if (bytes < 1024) return bytes + ' B';
            if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
            return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
        }
        
        removeFile.addEventListener('click', () => {
            selectedFile = null;
            fileInput.value = '';
            fileInfo.classList.remove('visible');
            processBtn.disabled = true;
        });
        
        processBtn.addEventListener('click', async () => {
            if (!selectedFile) return;
            
            processBtn.disabled = true;
            progressContainer.classList.add('visible');
            progressFill.style.width = '20%';
            progressText.textContent = 'Uploading document...';
            
            const formData = new FormData();
            formData.append('file', selectedFile);
            formData.append('word_count', document.getElementById('wordCount').value);
            formData.append('format_style', document.querySelector('input[name="formatStyle"]:checked').value);
            
            try {
                progressFill.style.width = '40%';
                progressText.textContent = 'Extracting incipits...';
                
                const response = await fetch('/process', {
                    method: 'POST',
                    body: formData
                });
                
                progressFill.style.width = '80%';
                progressText.textContent = 'Generating document...';
                
                if (response.ok) {
                    const blob = await response.blob();
                    const url = window.URL.createObjectURL(blob);
                    
                    progressFill.style.width = '100%';
                    progressText.textContent = 'Complete!';
                    
                    setTimeout(() => {
                        showResult(true, url, selectedFile.name);
                    }, 500);
                } else {
                    const error = await response.json();
                    showResult(false, null, null, error.error);
                }
            } catch (err) {
                showResult(false, null, null, err.message);
            }
        });
        
        function showResult(success, url, originalName, errorMsg) {
            uploadSection.style.display = 'none';
            resultSection.classList.add('visible');
            
            if (success) {
                const newName = originalName.replace('.docx', '_incipit.docx');
                resultContent.innerHTML = `
                    <div class="success-icon">✓</div>
                    <h2>Transformation Complete!</h2>
                    <p style="margin: 20px 0; color: #888;">Your document has been converted to incipit format.</p>
                    <a href="${url}" download="${newName}" class="btn btn-download">
                        Download ${newName}
                    </a>
                    <p style="margin-top: 20px; color: #666; font-size: 0.9rem;">
                        Remember: Open in Word, press Ctrl+A then F9 to update page numbers
                    </p>
                `;
            } else {
                resultContent.innerHTML = `
                    <div class="error-icon">✕</div>
                    <h2>Processing Failed</h2>
                    <p style="margin: 20px 0; color: #888;">${errorMsg || 'An error occurred while processing your document.'}</p>
                `;
            }
        }
        
        resetBtn.addEventListener('click', () => {
            selectedFile = null;
            fileInput.value = '';
            fileInfo.classList.remove('visible');
            processBtn.disabled = true;
            progressContainer.classList.remove('visible');
            progressFill.style.width = '0%';
            resultSection.classList.remove('visible');
            uploadSection.style.display = 'block';
        });
        
        // Radio button styling
        document.querySelectorAll('input[name="formatStyle"]').forEach(radio => {
            radio.addEventListener('change', () => {
                document.querySelectorAll('.radio-label').forEach(label => {
                    label.classList.remove('selected');
                });
                radio.closest('.radio-label').classList.add('selected');
            });
        });
    </script>
</body>
</html>
'''


@app.route('/')
def index():
    """Serve the main page."""
    return render_template_string(HTML_TEMPLATE)


@app.route('/process', methods=['POST'])
def process():
    """Process an uploaded document."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not file.filename.endswith('.docx'):
        return jsonify({'error': 'Please upload a .docx file'}), 400
    
    try:
        # Read the uploaded file
        docx_bytes = file.read()
        
        # Get options from form
        word_count = int(request.form.get('word_count', 3))
        format_style = request.form.get('format_style', 'bold')
        
        # Clamp word_count to valid range
        word_count = max(3, min(8, word_count))
        
        # Process the document
        transformed_bytes = process_document(docx_bytes, word_count=word_count, format_style=format_style)
        
        # Activate links (make URLs clickable)
        final_bytes = activate_links(transformed_bytes)
        
        # Generate output filename
        original_name = secure_filename(file.filename)
        output_name = original_name.replace('.docx', '_incipit.docx')
        
        # Return the processed file
        return send_file(
            BytesIO(final_bytes),
            mimetype='application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            as_attachment=True,
            download_name=output_name
        )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy'})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
