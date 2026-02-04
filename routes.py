import os
from flask import Blueprint, render_template, request, redirect, url_for, session, flash, send_file, jsonify
from werkzeug.utils import secure_filename
from auth import check_login
from model_handler import bone_model
import uuid
from datetime import datetime
import numpy as np

routes = Blueprint("routes", __name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
RESULTS_FOLDER = 'static/results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif', 'tiff'}

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@routes.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        if check_login(username, password):
            session["user"] = username
            return redirect(url_for("routes.analyze"))
        else:
            flash("Invalid credentials", "error")
            return redirect(url_for("routes.login"))

    return render_template("login.html")

@routes.route("/analyze", methods=["GET", "POST"])
def analyze():
    if "user" not in session:
        return redirect(url_for("routes.login"))
    
    # Check if model is loaded
    if not hasattr(bone_model, 'model') or bone_model.model is None:
        flash('Bone segmentation model is not loaded. Please check if the model file exists.', 'error')
    
    if request.method == "POST":
        # Check if file was uploaded
        if 'xray_file' not in request.files:
            flash('No file selected', 'error')
            return redirect(request.url)
        
        file = request.files['xray_file']
        
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            try:
                # Generate unique ID for this analysis
                analysis_id = str(uuid.uuid4())[:8]
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                
                # Save uploaded file
                filename = secure_filename(file.filename)
                unique_filename = f"{timestamp}_{analysis_id}_{filename}"
                upload_path = os.path.join(UPLOAD_FOLDER, unique_filename)
                file.save(upload_path)
                
                # Create results directory for this analysis
                result_dir = os.path.join(RESULTS_FOLDER, analysis_id)
                os.makedirs(result_dir, exist_ok=True)
                
                # Check if model is ready
                if not hasattr(bone_model, 'model') or bone_model.model is None:
                    flash('Model is not loaded. Please ensure model/bone_parts_segmentation.pth exists.', 'error')
                    return redirect(request.url)
                
                # Run bone segmentation
                pred_mask, original_img = bone_model.predict(upload_path)
                
                # Generate visualizations and get statistics
                results = bone_model.generate_visualizations(
                    original_img, 
                    pred_mask, 
                    result_dir, 
                    unique_filename
                )
                
                # Capture analysis notes/question
                analysis_notes = request.form.get('notes', '').strip()
                
                # Store results in session for display
                session['last_analysis'] = {
                    'analysis_id': analysis_id,
                    'original_image': f"uploads/{unique_filename}",
                    'mask_image': f"results/{analysis_id}/{os.path.basename(results['mask_path'])}",
                    'overlay_image': f"results/{analysis_id}/{os.path.basename(results['overlay_path'])}",
                    'statistics': results['statistics'],
                    'filename': filename,
                    'timestamp': timestamp,
                    'analysis_question': analysis_notes if analysis_notes else None
                }
                
                flash('Analysis completed successfully!', 'success')
                return redirect(url_for('routes.results', analysis_id=analysis_id))
                
            except Exception as e:
                flash(f'Error during analysis: {str(e)}', 'error')
                return redirect(request.url)
        else:
            flash('Invalid file type. Allowed: PNG, JPG, JPEG, TIF', 'error')
            return redirect(request.url)
    
    return render_template("analyze.html")

@routes.route("/results/<analysis_id>")
def results(analysis_id):
    if "user" not in session:
        return redirect(url_for("routes.login"))
    
    # Get analysis results from session
    if 'last_analysis' in session and session['last_analysis']['analysis_id'] == analysis_id:
        analysis_data = session['last_analysis']
        
        # Format statistics for display
        stats = analysis_data['statistics']
        
        # Vertebrae class mapping
        vertebrae_mapping = {
            'part_1': {'class': 'G', 'name': 'C5 Body'},
            'part_2': {'class': 'H', 'name': 'C5 Spinous Process'},
            'part_3': {'class': 'A', 'name': 'C1 Anterior Tubercle'},
            'part_4': {'class': 'B', 'name': 'Odontoid Process'},
            'part_5': {'class': 'C', 'name': 'C1 Posterior Arch'},
            'part_6': {'class': 'F', 'name': 'C2 Spinous Process'},
            'part_7': {'class': 'E', 'name': 'C5 Body'},
            'part_8': {'class': 'D', 'name': 'C1 Posterior Tubercle'}
        }
        
        # Create bone parts table data
        bone_parts = []
        if 'bone_parts' in stats:
            for part_name, part_data in stats['bone_parts'].items():
                mapping = vertebrae_mapping.get(part_name, {'class': 0, 'name': part_name.replace('_', ' ').title()})
                bone_parts.append({
                    'class': mapping['class'],
                    'name': mapping['name'],
                    'pixels': f"{part_data['pixels']:,}",
                    'percentage': f"{part_data['percentage']:.2f}%"
                })
            # Sort by class number
            bone_parts.sort(key=lambda x: x['class'])
        
        summary = {
            'filename': analysis_data['filename'],
            'timestamp': analysis_data['timestamp'],
            'image_size': f"{stats['image_size'][0]} x {stats['image_size'][1]}",
            'total_pixels': f"{stats['total_pixels']:,}",
            'bone_area': f"{stats['summary']['bone_percentage']:.2f}%",
            'bone_pixels': f"{stats['summary']['bone_pixels']:,}",
            'num_parts': stats['summary']['num_bone_parts'],
            'background': f"{stats['summary']['background_percentage']:.2f}%"
        }
        
        return render_template(
            "results.html",
            original_image=analysis_data['original_image'],
            mask_image=analysis_data['mask_image'],
            overlay_image=analysis_data['overlay_image'],
            bone_parts=bone_parts,
            summary=summary,
            analysis_id=analysis_id
        )
    
    flash('Analysis not found. Please upload a new X-ray image.', 'error')
    return redirect(url_for('routes.analyze'))

@routes.route("/download/<analysis_id>/<file_type>")
def download(analysis_id, file_type):
    if "user" not in session:
        return redirect(url_for("routes.login"))
    
    valid_types = ['mask', 'overlay', 'stats']
    if file_type not in valid_types:
        flash('Invalid file type', 'error')
        return redirect(url_for('routes.results', analysis_id=analysis_id))
    
    file_path = None
    filename = ""
    
    if 'last_analysis' in session and session['last_analysis']['analysis_id'] == analysis_id:
        result_dir = os.path.join(RESULTS_FOLDER, analysis_id)
        
        # Find files in result directory
        import glob
        if file_type == 'mask':
            file_pattern = os.path.join(result_dir, "mask_*.png")
            actual_files = glob.glob(file_pattern)
            if actual_files:
                filename = f"bone_segmentation_mask_{analysis_id}.png"
                return send_file(
                    actual_files[0],
                    as_attachment=True,
                    download_name=filename
                )
        elif file_type == 'overlay':
            file_pattern = os.path.join(result_dir, "overlay_*.png")
            actual_files = glob.glob(file_pattern)
            if actual_files:
                filename = f"bone_segmentation_overlay_{analysis_id}.png"
                return send_file(
                    actual_files[0],
                    as_attachment=True,
                    download_name=filename
                )
        elif file_type == 'stats':
            file_pattern = os.path.join(result_dir, "stats_*.json")
            actual_files = glob.glob(file_pattern)
            if actual_files:
                filename = f"bone_segmentation_stats_{analysis_id}.json"
                return send_file(
                    actual_files[0],
                    as_attachment=True,
                    download_name=filename
                )
    
    flash('File not found', 'error')
    return redirect(url_for('routes.results', analysis_id=analysis_id))

@routes.route("/model_status")
def model_status():
    """Check model status"""
    if "user" not in session:
        return redirect(url_for("routes.login"))
    
    status = {
        'loaded': hasattr(bone_model, 'model') and bone_model.model is not None,
        'device': str(bone_model.device) if hasattr(bone_model, 'device') else 'Unknown',
        'model_path': bone_model.model_path,
        'model_exists': os.path.exists(bone_model.model_path)
    }
    
    return jsonify(status)

@routes.route("/logout")
def logout():
    session.pop("user", None)
    session.pop("last_analysis", None)
    return redirect(url_for("routes.login"))