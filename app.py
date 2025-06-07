import os
import random
import json
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
from werkzeug.utils import secure_filename

# --- Configuration ---
# IMPORTANT: Change this to your actual image directory
# IMAGES_BASE_DIR = '/ssd_scratch/pratyush.jena/Aug_LineTR/GA_unified_v2/GA_unified_v2_Train/images_train'
# IMAGES_BASE_DIR = './sample_images' # For local testing if you create this folder
IMAGES_BASE_DIR = './images_train/'

ALLOWED_XCF_EXTENSIONS = {'xcf'}
ALLOWED_MASK_EXTENSIONS = {'png'} # Common mask formats

# Define annotator IDs and their quotas
ANNOTATOR_QUOTAS = {
    "Pratyush": 130,
    "Vaibhav": 50,
    "Amal": 50,
    "Abhinav": 50,
    "Aarnav": 48
}
ASSIGNMENTS_FILE = 'assignments.json'

app = Flask(__name__)
app.secret_key = 'your_very_secret_key_here' # Important for flashing messages
app.config['UPLOAD_FOLDER'] = IMAGES_BASE_DIR # Save uploads directly into the image folder

# --- Helper Functions ---
def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def get_image_files(directory):
    images = []
    valid_image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'}
    if not os.path.isdir(directory):
        print(f"Error: Image directory '{directory}' not found.")
        return []
    for f in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, f)) and \
           os.path.splitext(f)[1].lower() in valid_image_extensions:
            images.append(f)
    return images

def assign_images():
    """Assigns images to annotators or loads existing assignments."""
    if os.path.exists(ASSIGNMENTS_FILE):
        try:
            with open(ASSIGNMENTS_FILE, 'r') as f:
                assignments = json.load(f)
            print("Loaded existing image assignments.")
            # Verify that assigned files still exist
            all_assigned_files = set()
            for annotator_files in assignments.values():
                all_assigned_files.update(annotator_files)

            current_files = set(get_image_files(IMAGES_BASE_DIR))
            if not all_assigned_files.issubset(current_files):
                print("Warning: Some assigned files are missing. Re-assigning...")
                return create_new_assignments()
            return assignments
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading assignments file ({e}), creating new assignments.")
            return create_new_assignments()
    else:
        return create_new_assignments()

def create_new_assignments():
    print("Creating new image assignments...")
    all_images = get_image_files(IMAGES_BASE_DIR)
    if not all_images:
        print("No images found to assign.")
        return {annotator_id: [] for annotator_id in ANNOTATOR_QUOTAS.keys()}

    random.shuffle(all_images)
    assignments = {}
    current_index = 0
    total_quota = sum(ANNOTATOR_QUOTAS.values())

    if len(all_images) < total_quota:
        print(f"Warning: Only {len(all_images)} images available, but quota is {total_quota}.")
        # Adjust quotas proportionally or handle as an error - for now, assign what's available
    
    for annotator_id, quota in ANNOTATOR_QUOTAS.items():
        num_to_assign = min(quota, len(all_images) - current_index)
        assignments[annotator_id] = all_images[current_index : current_index + num_to_assign]
        current_index += num_to_assign
        if current_index >= len(all_images):
            # Fill remaining annotators with empty lists if out of images
            for remaining_annotator_id in list(ANNOTATOR_QUOTAS.keys())[list(ANNOTATOR_QUOTAS.keys()).index(annotator_id) + 1:]:
                assignments[remaining_annotator_id] = []


    # Save assignments
    try:
        with open(ASSIGNMENTS_FILE, 'w') as f:
            json.dump(assignments, f, indent=4)
        print(f"Saved new assignments to {ASSIGNMENTS_FILE}")
    except IOError as e:
        print(f"Error saving assignments: {e}")
    return assignments

# Initialize assignments when the app starts
IMAGE_ASSIGNMENTS = assign_images()

# --- Routes ---
@app.route('/')
def index():
    annotators = list(IMAGE_ASSIGNMENTS.keys())
    return render_template('index.html', annotators=annotators, annotators_names=ANNOTATOR_QUOTAS.keys())

@app.route('/annotator/<annotator_id>')
def annotator_page(annotator_id):
    if annotator_id not in IMAGE_ASSIGNMENTS:
        flash(f"Annotator ID '{annotator_id}' not found.", "error")
        return redirect(url_for('index'))

    assigned_images_filenames = IMAGE_ASSIGNMENTS.get(annotator_id, [])
    annotated_status = []

    for filename in assigned_images_filenames:
        base, orig_ext = os.path.splitext(filename)
        xcf_exists = os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], base + ".xcf"))
        
        # Check for any mask file (e.g., base_mask.png, base_mask.jpg)
        mask_exists = False
        for ext in ALLOWED_MASK_EXTENSIONS:
            if os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], base + "_mask." + ext)):
                mask_exists = True
                break
        
        annotated_status.append({
            "original": filename,
            "xcf_exists": xcf_exists,
            "mask_exists": mask_exists
        })

    return render_template('annotator_view.html',
                           annotator_id=annotator_id,
                           images_status=annotated_status)

@app.route('/download/<annotator_id>/<filename>')
def download_file(annotator_id, filename):
    # Security check: Ensure the requested file is actually assigned to this annotator
    if annotator_id not in IMAGE_ASSIGNMENTS or \
       filename not in IMAGE_ASSIGNMENTS[annotator_id]:
        flash("Error: You are not authorized to download this file or file not found.", "error")
        return redirect(url_for('annotator_page', annotator_id=annotator_id))

    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)
    except FileNotFoundError:
        flash(f"Error: File '{filename}' not found on server.", "error")
        return redirect(url_for('annotator_page', annotator_id=annotator_id))


@app.route('/upload/<annotator_id>/<original_filename>', methods=['POST'])
def upload_files(annotator_id, original_filename):
    if request.method == 'POST':
        # Security check: Ensure the original_filename is assigned to this annotator
        if annotator_id not in IMAGE_ASSIGNMENTS or \
           original_filename not in IMAGE_ASSIGNMENTS[annotator_id]:
            flash("Error: Invalid upload target.", "error")
            return redirect(url_for('annotator_page', annotator_id=annotator_id))

        base_filename, _ = os.path.splitext(original_filename)
        
        xcf_file = request.files.get('xcf_file')
        mask_file = request.files.get('mask_file')

        uploaded_xcf = False
        uploaded_mask = False

        if xcf_file and xcf_file.filename != '':
            if allowed_file(xcf_file.filename, ALLOWED_XCF_EXTENSIONS):
                # Save as original_basename.xcf
                xcf_savename = secure_filename(base_filename + ".xcf")
                xcf_savepath = os.path.join(app.config['UPLOAD_FOLDER'], xcf_savename)
                try:
                    xcf_file.save(xcf_savepath)
                    flash(f"XCF file '{xcf_savename}' uploaded successfully.", "success")
                    uploaded_xcf = True
                except Exception as e:
                    flash(f"Error saving XCF file: {e}", "error")
            else:
                flash("Invalid XCF file type. Allowed: .xcf", "error")
        elif 'xcf_file' in request.files and xcf_file.filename == '': # Field was present but empty
            pass # No XCF file uploaded, which is fine
        else: # Field was not even present
            flash("XCF file part missing in the form.", "warning")


        if mask_file and mask_file.filename != '':
            if allowed_file(mask_file.filename, ALLOWED_MASK_EXTENSIONS):
                mask_extension = mask_file.filename.rsplit('.', 1)[1].lower()
                # Save as original_basename_mask.ext
                mask_savename = secure_filename(base_filename + "_mask." + mask_extension)
                mask_savepath = os.path.join(app.config['UPLOAD_FOLDER'], mask_savename)
                try:
                    mask_file.save(mask_savepath)
                    flash(f"Mask file '{mask_savename}' uploaded successfully.", "success")
                    uploaded_mask = True
                except Exception as e:
                    flash(f"Error saving MASK file: {e}", "error")
            else:
                flash(f"Invalid mask file type. Allowed: {', '.join(ALLOWED_MASK_EXTENSIONS)}", "error")
        elif 'mask_file' in request.files and mask_file.filename == '':
            pass
        else:
            flash("Mask file part missing in the form.", "warning")

        if not uploaded_xcf and not uploaded_mask and \
           ( (xcf_file and xcf_file.filename != '') or \
             (mask_file and mask_file.filename != '') ): # attempted upload but failed
            pass # Errors already flashed
        elif not (xcf_file and xcf_file.filename != '') and not (mask_file and mask_file.filename != ''):
            flash("No files selected for upload.", "info")


        return redirect(url_for('annotator_page', annotator_id=annotator_id))

    # Should not happen for POST, but good practice
    return redirect(url_for('annotator_page', annotator_id=annotator_id))


if __name__ == '__main__':
    if not os.path.isdir(IMAGES_BASE_DIR):
        print(f"ERROR: The image directory '{IMAGES_BASE_DIR}' does not exist.")
        print("Please create it and add images, or update the 'IMAGES_BASE_DIR' variable in app.py.")
        # You might want to exit here if the directory is critical for startup
        # exit(1) 
        # For testing, we can create a dummy one if it's the sample_images path
        if IMAGES_BASE_DIR == './sample_images':
            print("Creating ./sample_images for testing...")
            os.makedirs(IMAGES_BASE_DIR, exist_ok=True)
            # Create some dummy image files for testing
            for i in range(10):
                with open(os.path.join(IMAGES_BASE_DIR, f'test_image_{i+1}.jpg'), 'w') as f:
                    f.write("dummy image content") # Not a real image, but file exists
            # Re-initialize assignments if we just created dummy files
            IMAGE_ASSIGNMENTS = assign_images()


    print(f"Serving images from: {os.path.abspath(IMAGES_BASE_DIR)}")
    print("Annotator assignments:")
    for ann, imgs in IMAGE_ASSIGNMENTS.items():
        print(f"  {ann}: {len(imgs)} images")

    app.run(debug=True, host='0.0.0.0', port=5002) # Accessible on your network