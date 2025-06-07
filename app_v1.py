import os
import random
import json
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash, Response
from werkzeug.utils import secure_filename
import cv2 # Added
import numpy as np # Added
import io # Added

# --- Configuration ---
# IMAGES_BASE_DIR = '/ssd_scratch/pratyush.jena/Aug_LineTR/GA_unified_v2/GA_unified_v2_Train/images_train'
# IMAGES_BASE_DIR = './sample_images' # For local testing
IMAGES_BASE_DIR = './images_train/'

ALLOWED_XCF_EXTENSIONS = {'xcf'}
ALLOWED_MASK_EXTENSIONS = {'png'}

# Define annotator IDs (names) and their quotas
ANNOTATOR_QUOTAS = {
    "Pratyush": 130,
    "Vaibhav": 50,
    "Amal": 50,
    "Abhinav": 50,
    "Arnav": 50
}
ASSIGNMENTS_FILE = 'assignments.json'

app = Flask(__name__)
app.secret_key = 'your_very_secret_key_here_CHANGE_ME'
app.config['UPLOAD_FOLDER'] = IMAGES_BASE_DIR

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
        if os.path.isfile(os.path.join(directory, f)) and os.path.splitext(f)[1].lower() in valid_image_extensions:
            # check if _mask is not in the filename
            if '_mask' not in f and not f.startswith('.'):
                images.append(f)
    return images

def assign_images():
    if os.path.exists(ASSIGNMENTS_FILE):
        try:
            with open(ASSIGNMENTS_FILE, 'r') as f:
                assignments = json.load(f)
            print("Loaded existing image assignments.")
            all_assigned_files = set()
            for annotator_files in assignments.values(): # assignments keys are annotator names
                all_assigned_files.update(annotator_files)
            current_files = set(get_image_files(IMAGES_BASE_DIR))
            if not all_assigned_files.issubset(current_files):
                print("Warning: Some assigned files are missing or annotator list changed. Re-assigning...")
                return create_new_assignments()
            # Check if annotator keys match
            if set(assignments.keys()) != set(ANNOTATOR_QUOTAS.keys()):
                print("Warning: Annotator list in assignments.json differs from ANNOTATOR_QUOTAS. Re-assigning...")
                return create_new_assignments()
            return assignments
        except (json.JSONDecodeError, IOError, KeyError) as e:
            print(f"Error loading assignments file ({e}), creating new assignments.")
            return create_new_assignments()
    else:
        return create_new_assignments()

# --- Helper function to calculate stats (can be inside app.py or imported) ---
def get_annotator_stats(annotator_name, assigned_images):
    completed_count = 0
    partial_xcf_count = 0 # Only XCF uploaded
    partial_mask_count = 0 # Only Mask uploaded
    pending_count = 0
    total_assigned = len(assigned_images)

    for filename in assigned_images:
        base, _ = os.path.splitext(filename)
        xcf_exists = os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], base + ".xcf"))
        mask_file_path = find_mask_file_path(app.config['UPLOAD_FOLDER'], base) # Uses existing helper
        mask_exists = mask_file_path is not None

        if xcf_exists and mask_exists:
            completed_count += 1
        elif xcf_exists and not mask_exists:
            partial_xcf_count += 1
        elif not xcf_exists and mask_exists:
            partial_mask_count += 1
        else: # Neither exists
            pending_count += 1
    
    return {
        "total": total_assigned,
        "completed": completed_count,
        "partial_xcf": partial_xcf_count,
        "partial_mask": partial_mask_count,
        "pending": pending_count
    }

def create_new_assignments():
    print("Creating new image assignments...")
    all_images = get_image_files(IMAGES_BASE_DIR)
    if not all_images:
        print("No images found to assign.")
        return {annotator_name: [] for annotator_name in ANNOTATOR_QUOTAS.keys()}

    random.seed(42)
    random.shuffle(all_images)
    assignments = {}
    current_index = 0
    total_quota = sum(ANNOTATOR_QUOTAS.values())

    if len(all_images) < total_quota:
        print(f"Warning: Only {len(all_images)} images available, but total quota is {total_quota}.")

    for annotator_name, quota in ANNOTATOR_QUOTAS.items():
        num_to_assign = min(quota, len(all_images) - current_index)
        assignments[annotator_name] = all_images[current_index : current_index + num_to_assign]
        current_index += num_to_assign
        if current_index >= len(all_images) and list(ANNOTATOR_QUOTAS.keys()).index(annotator_name) < len(ANNOTATOR_QUOTAS) -1:
            for remaining_annotator_id in list(ANNOTATOR_QUOTAS.keys())[list(ANNOTATOR_QUOTAS.keys()).index(annotator_name) + 1:]:
                assignments[remaining_annotator_id] = []


    try:
        with open(ASSIGNMENTS_FILE, 'w') as f:
            json.dump(assignments, f, indent=4)
        print(f"Saved new assignments to {ASSIGNMENTS_FILE}")
    except IOError as e:
        print(f"Error saving assignments: {e}")
    return assignments

IMAGE_ASSIGNMENTS = assign_images()

def find_mask_file_path(base_dir, original_filename_base):
    """Finds an existing mask file for the given original image base name."""
    for ext in ALLOWED_MASK_EXTENSIONS:
        mask_filename = f"{original_filename_base}_mask.{ext}"
        # print(mask_filename) 
        mask_path = os.path.join(base_dir, mask_filename)
        if os.path.exists(mask_path):
            return mask_path
    return None

def serve_pil_image(pil_img, image_format='PNG'):
    """Serves a PIL/Pillow image object as a Flask response."""
    img_io = io.BytesIO()
    pil_img.save(img_io, image_format)
    img_io.seek(0)
    return Response(img_io.getvalue(), mimetype=f'image/{image_format.lower()}')

def serve_cv_image(cv_image, image_format_ext='.png'):
    """Serves an OpenCV image (NumPy array) as a Flask response."""
    is_success, buffer = cv2.imencode(image_format_ext, cv_image)
    if is_success:
        return Response(io.BytesIO(buffer).getvalue(), mimetype=f'image/{image_format_ext.strip(".")}')
    else:
        flash("Error encoding image for display.", "error")
        return "Error encoding image", 500


# --- Main Routes ---
@app.route('/')
def index():
    annotator_info_list = []
    for annotator_name, assigned_images in IMAGE_ASSIGNMENTS.items():
        stats = get_annotator_stats(annotator_name, assigned_images)
        annotator_info_list.append({
            "name": annotator_name,
            "stats": stats
        })
    return render_template('index.html', annotators_info=annotator_info_list)

@app.route('/annotator/<annotator_name>') # Changed from annotator_id to annotator_name
def annotator_page(annotator_name):
    if annotator_name not in IMAGE_ASSIGNMENTS:
        flash(f"Annotator '{annotator_name}' not found.", "error")
        return redirect(url_for('index'))

    assigned_images_filenames = IMAGE_ASSIGNMENTS.get(annotator_name, [])
    annotated_status = []

    for filename in assigned_images_filenames:
        base, orig_ext = os.path.splitext(filename)
        xcf_exists = os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], base + ".xcf"))
        mask_file_path = find_mask_file_path(app.config['UPLOAD_FOLDER'], base)
        mask_exists = mask_file_path is not None

        annotated_status.append({
            "original": filename,
            "xcf_exists": xcf_exists,
            "mask_exists": mask_exists,
            "base_filename": base # For constructing view URLs
        })

    return render_template('annotator_view.html',
                           annotator_name=annotator_name,
                           images_status=annotated_status)

@app.route('/download/<annotator_name>/<filename>')
def download_file(annotator_name, filename):
    if annotator_name not in IMAGE_ASSIGNMENTS or \
       filename not in IMAGE_ASSIGNMENTS[annotator_name]:
        flash("Error: You are not authorized to download this file or file not found.", "error")
        return redirect(url_for('annotator_page', annotator_name=annotator_name))
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)
    except FileNotFoundError:
        flash(f"Error: File '{filename}' not found on server.", "error")
        return redirect(url_for('annotator_page', annotator_name=annotator_name))

@app.route('/upload/<annotator_name>/<original_filename>', methods=['POST'])
def upload_files(annotator_name, original_filename):
    if request.method == 'POST':
        if annotator_name not in IMAGE_ASSIGNMENTS or \
           original_filename not in IMAGE_ASSIGNMENTS[annotator_name]:
            flash("Error: Invalid upload target.", "error")
            return redirect(url_for('annotator_page', annotator_name=annotator_name))

        base_filename, _ = os.path.splitext(original_filename)
        xcf_file = request.files.get('xcf_file')
        mask_file = request.files.get('mask_file')
        # ... (rest of the upload logic is the same, ensure it uses annotator_name) ...
        uploaded_xcf = False
        uploaded_mask = False

        if xcf_file and xcf_file.filename != '':
            if allowed_file(xcf_file.filename, ALLOWED_XCF_EXTENSIONS):
                # xcf_savename = secure_filename(base_filename + ".xcf")
                xcf_savename = base_filename + ".xcf"
                print(f"Saving XCF as: {xcf_savename}") # Debug print
                xcf_savepath = os.path.join(app.config['UPLOAD_FOLDER'], xcf_savename)
                try:
                    xcf_file.save(xcf_savepath)
                    flash(f"XCF file '{xcf_savename}' uploaded successfully.", "success")
                    uploaded_xcf = True
                except Exception as e:
                    flash(f"Error saving XCF file: {e}", "error")
            else:
                flash("Invalid XCF file type. Allowed: .xcf", "error")
        elif 'xcf_file' in request.files and xcf_file.filename == '':
            pass
        else:
            flash("XCF file part missing in the form.", "warning")


        if mask_file and mask_file.filename != '':
            if allowed_file(mask_file.filename, ALLOWED_MASK_EXTENSIONS):
                mask_extension = mask_file.filename.rsplit('.', 1)[1].lower()
                # mask_savename = secure_filename(base_filename + "_mask." + mask_extension)
                mask_savename = base_filename + "_mask." + mask_extension
                print(f"Saving mask as: {mask_savename}") # Debug print
                mask_savepath = os.path.join(app.config['UPLOAD_FOLDER'], mask_savename)
                print(f"Mask save path: {mask_savepath}") # Debug print
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
             (mask_file and mask_file.filename != '') ):
            pass
        elif not (xcf_file and xcf_file.filename != '') and not (mask_file and mask_file.filename != ''):
            flash("No files selected for upload.", "info")

        return redirect(url_for('annotator_page', annotator_name=annotator_name))
    return redirect(url_for('annotator_page', annotator_name=annotator_name))


# --- Image Viewing Routes ---
def get_paths_for_view(annotator_name, original_filename_with_ext):
    if annotator_name not in IMAGE_ASSIGNMENTS or \
       original_filename_with_ext not in IMAGE_ASSIGNMENTS[annotator_name]:
        return None, None, "Authorization error or file not assigned."

    original_img_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename_with_ext)
    if not os.path.exists(original_img_path):
        return None, None, "Original image not found."

    original_filename_base, _ = os.path.splitext(original_filename_with_ext)
    mask_img_path = find_mask_file_path(app.config['UPLOAD_FOLDER'], original_filename_base)

    return original_img_path, mask_img_path, None


@app.route('/view/original/<annotator_name>/<original_filename_with_ext>')
def view_original_image(annotator_name, original_filename_with_ext):
    original_img_path, _, error_msg = get_paths_for_view(annotator_name, original_filename_with_ext)
    if error_msg:
        flash(error_msg, "error")
        return error_msg, 404 # Or redirect, or a placeholder image

    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], original_filename_with_ext)
    except FileNotFoundError:
        return "Original image file not found on server.", 404

@app.route('/view/mask/<annotator_name>/<original_filename_with_ext>')
def view_mask_image(annotator_name, original_filename_with_ext):
    _, mask_img_path, error_msg = get_paths_for_view(annotator_name, original_filename_with_ext)
    if error_msg:
        flash(error_msg, "error")
        return error_msg, 404
    if not mask_img_path:
        return "Mask image not found.", 404

    try:
        return send_from_directory(os.path.dirname(mask_img_path), os.path.basename(mask_img_path))
    except FileNotFoundError:
        return "Mask image file not found on server.", 404


@app.route('/view/binary_mask/<annotator_name>/<original_filename_with_ext>')
def view_binary_mask_image(annotator_name, original_filename_with_ext):
    _, mask_img_path, error_msg = get_paths_for_view(annotator_name, original_filename_with_ext)
    if error_msg:
        flash(error_msg, "error")
        return error_msg, 404
    if not mask_img_path:
        return "Mask image not found to create binary version.", 404

    try:
        img = cv2.imread(mask_img_path)
        if img is None:
            return "Could not read mask image.", 500
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
        return serve_cv_image(binary)

    except Exception as e:
        print(f"Error processing binary mask: {e}")
        return f"Error generating binary mask: {e}", 500

@app.route('/view/overlay/<annotator_name>/<original_filename_with_ext>')
def view_overlay_image(annotator_name, original_filename_with_ext):
    original_img_path, mask_img_path, error_msg = get_paths_for_view(annotator_name, original_filename_with_ext)
    if error_msg:
        flash(error_msg, "error")
        return error_msg, 404
    if not mask_img_path:
        return "Mask image not found for overlay.", 404
    if not original_img_path: # Should be caught by get_paths_for_view already
        return "Original image not found for overlay.", 404

    try:
        og_img = cv2.imread(original_img_path)
        mask_img_cv = cv2.imread(mask_img_path)

        if og_img is None: return "Could not read original image for overlay.", 500
        if mask_img_cv is None: return "Could not read mask image for overlay.", 500

        og_img = cv2.cvtColor(og_img, cv2.COLOR_BGR2RGB) # To RGB for consistency with snippet

        # Binarize the mask
        gray_mask = cv2.cvtColor(mask_img_cv, cv2.COLOR_BGR2GRAY)
        _, binary_mask_cv = cv2.threshold(gray_mask, 30, 255, cv2.THRESH_BINARY)

        # Create a colored mask (blue)
        # Ensure og_img and color_mask are same size
        if og_img.shape[:2] != binary_mask_cv.shape[:2]:
             # Resize binary_mask_cv to match og_img, or vice-versa.
             # Sane default: resize mask to original image.
             binary_mask_cv = cv2.resize(binary_mask_cv, (og_img.shape[1], og_img.shape[0]), interpolation=cv2.INTER_NEAREST)


        color_mask = cv2.cvtColor(binary_mask_cv, cv2.COLOR_GRAY2RGB)
        color_mask[np.where((color_mask == [255, 255, 255]).all(axis=2))] = [0, 0, 255] # Blue

        # Ensure og_img and color_mask are same size before addWeighted
        if og_img.shape != color_mask.shape:
            # This case might happen if original is color and mask was processed to different depth
            # For simplicity, if depths differ but H,W are same, proceed.
            # A more robust solution might involve resizing or erroring.
            # For now, let's assume cv2.addWeighted handles minor type differences or we ensure types match
            print(f"Warning: Shape mismatch for overlay. Original: {og_img.shape}, Color Mask: {color_mask.shape}")
            # Attempt to make them compatible if just channel depth differs
            if og_img.shape[:2] == color_mask.shape[:2] and og_img.ndim != color_mask.ndim:
                 if og_img.ndim == 2 and color_mask.ndim == 3: # og_img is grayscale
                     og_img = cv2.cvtColor(og_img, cv2.COLOR_GRAY2RGB)
                 # Add other conversion cases if necessary
            if og_img.shape != color_mask.shape: # If still not matching
                return "Image and mask dimensions are incompatible for overlay after processing.", 500


        overlay = cv2.addWeighted(og_img, 0.7, color_mask, 0.3, 0) # Adjusted weights for better visibility
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR) # Convert back to BGR for cv2.imencode

        return serve_cv_image(overlay_bgr)

    except Exception as e:
        print(f"Error processing overlay: {e}")
        return f"Error generating overlay: {e}", 500


if __name__ == '__main__':
    if not os.path.isdir(IMAGES_BASE_DIR):
        print(f"ERROR: The image directory '{IMAGES_BASE_DIR}' does not exist.")
        if IMAGES_BASE_DIR == './sample_images':
            print("Creating ./sample_images for testing...")
            os.makedirs(IMAGES_BASE_DIR, exist_ok=True)
            for i in range(10):
                # Create dummy JPG files for testing view routes
                dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
                cv2.imwrite(os.path.join(IMAGES_BASE_DIR, f'test_image_{i+1}.jpg'), dummy_img)
            IMAGE_ASSIGNMENTS = assign_images() # Re-assign after creating files
        else:
            print("Please create it and add images, or update the 'IMAGES_BASE_DIR' variable in app.py.")
            # exit(1) # Consider exiting if critical

    print(f"Serving images from: {os.path.abspath(IMAGES_BASE_DIR)}")
    print("Annotator assignments:")
    for ann, imgs in IMAGE_ASSIGNMENTS.items():
        print(f"  {ann}: {len(imgs)} images")

    app.run(debug=True, host='0.0.0.0', port=5000)