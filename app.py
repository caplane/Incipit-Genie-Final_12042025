"""
app.py

Flask application for Incipit Genie.

Transforms Word documents with endnotes into Word documents with incipit notes.
"""

import os
from flask import Flask, request, send_file, render_template, jsonify
from werkzeug.utils import secure_filename
from io import BytesIO

from document_processor import process_document
from link_activator import activate_links

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload




@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')


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
