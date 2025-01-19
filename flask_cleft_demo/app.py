import os
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, IntegerField, SelectField
from wtforms.validators import InputRequired, Length, EqualTo, Optional
from flask_wtf.file import FileField, FileAllowed
from services.ml_interface import process_image
from services.face_detection import detect_faces
from werkzeug.utils import secure_filename

from flask import Flask, render_template, Response
import cv2
import dlib
from flask_migrate import Migrate

app = Flask(__name__)

# 获取当前文件所在目录的绝对路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 构建模型文件的完整路径
MODEL_PATH = os.path.join(BASE_DIR, 'services', 'shape_predictor_68_face_landmarks.dat')

# 添加路径检查代码
def check_model_file():
    """检查模型文件是否存在并可访问"""
    print(f"正在检查模型文件路径: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"找不到模型文件: {MODEL_PATH}")
    if not os.access(MODEL_PATH, os.R_OK):
        raise PermissionError(f"无法读取模型文件: {MODEL_PATH}")
    print("模型文件检查通过！")

# 修改模型加载部分
try:
    check_model_file()
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(MODEL_PATH)
    print("模型加载成功！")
except Exception as e:
    print(f"模型加载失败: {str(e)}")
    # 这里可以选择是否继续运行
    # raise e  # 如果想在模型加载失败时停止运行，取消这行的注释

def generate_video():
    """
    视频流生成器，实时处理每帧图像，检测人脸并绘制 landmarks
    """
    video_capture = cv2.VideoCapture(0)

    while True:
        success, frame = video_capture.read()
        if not success:
            break

        # 转为灰度图并检测人脸
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        # 在检测到的人脸上绘制 landmarks
        for face in faces:
            landmarks = predictor(gray, face)
            for n in range(68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        # 编码为 JPEG 格式
        _, buffer = cv2.imencode('.jpg', frame)
        frame_data = buffer.tobytes()

        # 使用流式方式返回图像数据
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')

    video_capture.release()


# --- Flask Setup ---
app.config['SECRET_KEY'] = 'replace-with-a-strong-secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'  # Local SQLite file
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

migrate = Migrate(app, db)

# --- Database Models ---
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(10), nullable=False)  # 'patient' or 'doctor'
    cases = db.relationship('Case', backref='user', lazy=True)
    profile = db.relationship('UserProfile', uselist=False, back_populates='user')

class Case(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    pre_image = db.Column(db.Text, nullable=True)
    post_image = db.Column(db.Text, nullable=True)
    pre_severity = db.Column(db.String(50), nullable=True)
    post_severity = db.Column(db.String(50), nullable=True)
    pre_ratio = db.Column(db.Float)
    post_ratio = db.Column(db.Float)
    doctor_reviewed = db.Column(db.Boolean, default=False)
    doctor_approved = db.Column(db.Boolean, default=None)

class UserProfile(db.Model):
    __tablename__ = 'user_profile'  # 明确指定表名
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False, unique=True)
    age = db.Column(db.Integer)
    gender = db.Column(db.String(10))
    contact = db.Column(db.String(100))
    user = db.relationship('User', back_populates='profile')

# --- WTForms ---
class RegisterForm(FlaskForm):
    username = StringField('Username', validators=[InputRequired(), Length(min=4, max=25)])
    password = PasswordField('Password', validators=[InputRequired(), Length(min=4, max=25)])
    confirm = PasswordField('Confirm Password', validators=[EqualTo('password', message='Passwords must match')])
    doctor_code = StringField('Doctor Code (for Doctors)', validators=[Length(max=10)])
    submit = SubmitField('Sign Up')

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[InputRequired(), Length(min=4, max=25)])
    password = PasswordField('Password', validators=[InputRequired(), Length(min=4, max=25)])
    submit = SubmitField('Login')

class ProfileForm(FlaskForm):
    age = IntegerField('Age', validators=[Optional()])
    gender = SelectField('Gender', 
                        choices=[('', 'Select Gender'), 
                                ('Male', 'Male'), 
                                ('Female', 'Female'), 
                                ('Other', 'Other')],
                        validators=[Optional()])
    contact = StringField('Contact', validators=[Optional()])
    submit = SubmitField('Update Profile')

# --- Utility Functions ---
def init_db():
    """Create the database tables if they don't exist."""
    with app.app_context():
        # 检查数据库文件是否存在
        db_path = os.path.join(app.instance_path, 'database.db')
        if not os.path.exists(db_path):
            # 确保 instance 文件夹存在
            if not os.path.exists(app.instance_path):
                os.makedirs(app.instance_path)
            # 只在数据库不存在时创建表
            db.create_all()
            print("Database initialized successfully.")
        else:
            print("Database already exists.")

# --- Routes ---
@app.route('/')
def home():
    if 'user_id' in session:
        if session.get('role') == 'doctor':
            return redirect(url_for('view_all_cases'))
        else:
            return redirect(url_for('view_my_cases'))
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()
    if form.validate_on_submit():
        existing_user = User.query.filter_by(username=form.username.data).first()
        if existing_user:
            flash("Username already exists. Please choose a different username.")
            return redirect(url_for('register'))

        role = 'patient'
        if form.doctor_code.data == '1234':
            role = 'doctor'
        elif form.doctor_code.data:
            flash("Invalid doctor code.")
            return redirect(url_for('register'))

        hashed_password = generate_password_hash(form.password.data, method='pbkdf2:sha256')
        new_user = User(username=form.username.data, password_hash=hashed_password, role=role)
        db.session.add(new_user)
        db.session.commit()

        flash("Registration successful. Please log in.")
        return redirect(url_for('login'))
    return render_template('register.html', form=form)

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and check_password_hash(user.password_hash, form.password.data):
            session['user_id'] = user.id
            session['role'] = user.role
            session['username'] = user.username
            flash("Login successful!")
            return redirect(url_for('home'))
        else:
            flash("Invalid username or password.")
            return redirect(url_for('login'))
    return render_template('login.html', form=form)


@app.route('/logout')
def logout():
    session.clear()  # Clear all session data
    flash("Logged out successfully.")
    return redirect(url_for('login'))


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    """
    Single-page upload for pre-op and post-op images.
    Displays processed images, severity, and ratio after upload.
    """
    if 'user_id' not in session:
        flash("Please log in to access the upload page.")
        return redirect(url_for('login'))

    pre_result = None
    post_result = None

    if request.method == 'POST':
        # Process pre-op image
        if 'pre_op_image' in request.files and request.files['pre_op_image'].filename:
            pre_base64, pre_ratio, pre_severity = process_image(request.files['pre_op_image'])
            pre_result = (pre_base64, pre_ratio, pre_severity)

        # Process post-op image
        if 'post_op_image' in request.files and request.files['post_op_image'].filename:
            post_base64, post_ratio, post_severity = process_image(request.files['post_op_image'])
            post_result = (post_base64, post_ratio, post_severity)



    return render_template('upload.html', pre_result=pre_result, post_result=post_result)

@app.route('/video_feed')
def video_feed():
    """
    返回视频流响应，用于显示实时视频
    """
    return Response(generate_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/camera')
def camera():
    """
    渲染摄像头页面
    """
    return render_template('camera.html')

@app.route('/face-detection', methods=['POST'])
def face_detection():
    """
    接收前端的 Base64 图像并进行人脸检测
    """
    data = request.json.get('image')
    if not data:
        return jsonify({'error': 'No image data received'}), 400

    # 解码 Base64 图像
    image_data = base64.b64decode(data.split(',')[1])
    np_arr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # 检测人脸并绘制 landmarks
    faces, annotated_img = detect_faces(img)

    # 将处理后的图像编码为 Base64
    _, buffer = cv2.imencode('.png', annotated_img)
    encoded_img = base64.b64encode(buffer).decode('utf-8')

    return jsonify({'faces_detected': len(faces), 'image': encoded_img})



@app.route('/cases', methods=['GET'])
def view_my_cases():
    """
    Route for patients to view their own cases.
    """
    if 'user_id' not in session:
        flash("Please log in to view your cases.")
        return redirect(url_for('login'))

    user_id = session['user_id']
    cases = Case.query.filter_by(user_id=user_id).all()  # Retrieve cases for the logged-in user
    return render_template('cases.html', cases=cases)


@app.route('/cases/add', methods=['GET', 'POST'])
def add_case():
    if 'user_id' not in session:
        flash("Please log in to add a case.")
        return redirect(url_for('login'))

    if request.method == 'POST':
        pre_image = request.files.get('pre_op_image')
        post_image = request.files.get('post_op_image')

        if not (pre_image and pre_image.filename) and not (post_image and post_image.filename):
            flash("Please upload at least one image to add a case.")
            return redirect(url_for('add_case'))

        pre_result = None
        post_result = None

        if pre_image and pre_image.filename:
            pre_base64, pre_ratio, pre_severity = process_image(pre_image)
            pre_result = (pre_base64, pre_ratio, pre_severity)

        if post_image and post_image.filename:
            post_base64, post_ratio, post_severity = process_image(post_image)
            post_result = (post_base64, post_ratio, post_severity)

        new_case = Case(
            user_id=session['user_id'],
            pre_image=pre_result[0] if pre_result else None,
            post_image=post_result[0] if post_result else None,
            pre_severity=pre_result[2] if pre_result else None,
            post_severity=post_result[2] if post_result else None,
            pre_ratio=pre_result[1] if pre_result else None,
            post_ratio=post_result[1] if post_result else None,
        )
        db.session.add(new_case)
        db.session.commit()

        flash("Case added successfully!")
        return redirect(url_for('view_my_cases'))
        
    return render_template('add_case.html')  # 使用新的模板

@app.route('/cases/add_for_patient', methods=['GET', 'POST'])
def add_case_for_patient():
    if 'user_id' not in session:
        flash("Please log in to add a case.")
        return redirect(url_for('login'))

    user = User.query.get(session['user_id'])
    if user.role != 'doctor':
        flash("You do not have permission to access this page.")
        return redirect(url_for('home'))

    patients = User.query.filter_by(role='patient').all()

    if request.method == 'POST':
        selected_patient_id = request.form.get('patient_id')
        pre_image = request.files.get('pre_op_image')
        post_image = request.files.get('post_op_image')

        if not selected_patient_id:
            flash("Please select a patient to add a case.")
            return redirect(url_for('add_case_for_patient'))

        if not (pre_image and pre_image.filename) and not (post_image and post_image.filename):
            flash("Please upload at least one image to add a case.")
            return redirect(url_for('add_case_for_patient'))

        pre_result = None
        post_result = None

        if pre_image and pre_image.filename:
            pre_base64, pre_ratio, pre_severity = process_image(pre_image)
            pre_result = (pre_base64, pre_ratio, pre_severity)

        if post_image and post_image.filename:
            post_base64, post_ratio, post_severity = process_image(post_image)
            post_result = (post_base64, post_ratio, post_severity)

        new_case = Case(
            user_id=selected_patient_id,
            pre_image=pre_result[0] if pre_result else None,
            post_image=post_result[0] if post_result else None,
            pre_severity=pre_result[2] if pre_result else None,
            post_severity=post_result[2] if post_result else None,
            pre_ratio=pre_result[1] if pre_result else None,
            post_ratio=post_result[1] if post_result else None,
        )
        db.session.add(new_case)
        db.session.commit()

        flash("Case added successfully for the selected patient!")
        return redirect(url_for('view_all_cases'))

    return render_template('add_case_for_patient.html', patients=patients)


@app.route('/cases/all', methods=['GET'])
def view_all_cases():
    if 'user_id' not in session:
        flash("Please log in to access this page.")
        return redirect(url_for('login'))

    user = User.query.get(session['user_id'])
    if not user:  # Handle case where user does not exist
        flash("User not found. Please log in again.")
        return redirect(url_for('login'))

    if user.role != 'doctor':
        flash("Access denied. Only doctors can view all cases.")
        return redirect(url_for('home'))

    cases = Case.query.all()
    return render_template('all_cases.html', cases=cases)

@app.route('/cases/delete/<int:case_id>', methods=['POST'])
def delete_case(case_id):
    if 'user_id' not in session:
        flash("Please log in to delete cases.")
        return redirect(url_for('login'))

    case = Case.query.get_or_404(case_id)

    # 检查是否有权限删除病例
    user = User.query.get(session['user_id'])
    if user.role != 'doctor' and case.user_id != session['user_id']:
        flash("You do not have permission to delete this case.")
        return redirect(url_for('view_my_cases'))

    # 删除病例
    db.session.delete(case)
    db.session.commit()
    flash("Case deleted successfully.")
    if user.role == 'doctor':
        return redirect(url_for('view_all_cases'))
    else:
        return redirect(url_for('view_my_cases'))


@app.route('/cases/review/<int:case_id>', methods=['GET', 'POST'])
def review_case(case_id):
    if 'user_id' not in session:
        flash("Please log in to access this page.")
        return redirect(url_for('login'))

    user = User.query.get(session['user_id'])
    if not user or user.role != 'doctor':
        flash("Access denied. Only doctors can review cases.")
        return redirect(url_for('home'))

    case = Case.query.get(case_id)
    if not case:
        flash("Case not found.")
        return redirect(url_for('view_all_cases'))

    if request.method == 'POST':
        if 'approve' in request.form:
            case.doctor_reviewed = True
            case.doctor_approved = True
            flash("Case approved successfully.")
        elif 'reject' in request.form:
            case.doctor_reviewed = True
            case.doctor_approved = False
            flash("Case rejected successfully.")
        db.session.commit()
        return redirect(url_for('view_all_cases'))

    return render_template('review_case.html', case=case)


@app.route('/statistics', methods=['GET'])
def statistics():
    if 'user_id' not in session:
        flash("Please log in to access this page.")
        return redirect(url_for('login'))

    user = User.query.get(session['user_id'])
    if user.role != 'doctor':
        flash("Access denied.")
        return redirect(url_for('home'))

    total_cases = Case.query.count()
    correct_cases = Case.query.filter_by(doctor_approved=True).count()
    accuracy = (correct_cases / total_cases * 100) if total_cases else 0

    return render_template('statistics.html', total_cases=total_cases, correct_cases=correct_cases, accuracy=accuracy)

@app.route('/process_image', methods=['POST'])
def process_image_route():
    if 'pre_op_image' not in request.files and 'post_op_image' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'})
    
    try:
        # 确定是pre还是post图片
        if 'pre_op_image' in request.files:
            file = request.files['pre_op_image']
            image_type = 'pre'
        else:
            file = request.files['post_op_image']
            image_type = 'post'
            
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
            
        # 直接处理上传的文件对象
        base64_str, ratio, severity = process_image(file)
        
        return jsonify({
            'success': True,
            'image': base64_str,
            'ratio': f"{ratio:.3f}",
            'severity': severity,
            'type': image_type
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if 'user_id' not in session:
        flash("Please log in to access your profile.")
        return redirect(url_for('login'))
    
    user = User.query.get(session['user_id'])
    if not user:
        flash("User not found.")
        return redirect(url_for('login'))
    
    # 获取或创建用户资料
    user_profile = user.profile
    if not user_profile:
        user_profile = UserProfile(user_id=user.id)
        db.session.add(user_profile)
        db.session.commit()
    
    form = ProfileForm()
    
    if form.validate_on_submit():
        user_profile.age = form.age.data
        user_profile.gender = form.gender.data
        user_profile.contact = form.contact.data
        db.session.commit()
        flash('Profile updated successfully!')
        return redirect(url_for('profile'))
    
    # 预填表单数据
    elif request.method == 'GET':
        form.age.data = user_profile.age
        form.gender.data = user_profile.gender
        form.contact.data = user_profile.contact
    
    return render_template('profile.html', 
                         form=form, 
                         user=user, 
                         profile=user_profile)

@app.route('/patient/<int:user_id>/profile')
def view_patient_profile(user_id):
    if 'user_id' not in session:
        flash("Please log in to view patient profiles.")
        return redirect(url_for('login'))
    
    # 检查当前用户是否是医生
    current_user = User.query.get(session['user_id'])
    if not current_user or current_user.role != 'doctor':
        flash("Only doctors can view patient profiles.")
        return redirect(url_for('home'))
    
    # 获取病人信息
    patient = User.query.get_or_404(user_id)
    if patient.role != 'patient':
        flash("Invalid patient ID.")
        return redirect(url_for('view_all_cases'))
    
    # 获取病人的所有病例
    cases = Case.query.filter_by(user_id=user_id).all()
    
    return render_template('patient_profile.html', 
                         patient=patient,
                         cases=cases)

# --- Initialize Database ---
if __name__ == '__main__':
    init_db()  # 现在这个函数只会在数据库不存在时创建表
    app.run(debug=True)
