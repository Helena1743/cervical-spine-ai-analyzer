from flask import Flask
from routes import routes
import os

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this to a secure secret key

# Register blueprints
app.register_blueprint(routes)

# Configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB limit

# Debug: Check paths
print("=" * 50)
print("STARTUP DEBUG INFORMATION")
print("=" * 50)
print("Current working directory:", os.getcwd())
model_path = 'model/bone_parts_segmentation.pth'
print("Model path:", model_path)
print("Model exists:", os.path.exists(model_path))
print("Full model path:", os.path.abspath(model_path))
print("=" * 50)

# Initialize model at startup (not using before_first_request)
with app.app_context():
    print("Initializing Bone Segmentation Model...")
    try:
        from model_handler import bone_model
        # Update the model path to be absolute
        bone_model.model_path = os.path.abspath(model_path)
        if bone_model.load_model():
            print("✓ Model initialized successfully")
        else:
            print("✗ Model initialization failed")
    except Exception as e:
        print(f"✗ Error during model initialization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs('static/results', exist_ok=True)
    os.makedirs('model', exist_ok=True)  # Ensure model directory exists
    
    app.run(debug=True, port=5000)