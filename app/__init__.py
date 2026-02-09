from flask import Flask
from app.config import config
from app.services.data_service import data_service
import logging
from logging.handlers import RotatingFileHandler
import os
from datetime import datetime

def create_app():
    # Explicitly set template and static folders using absolute paths
    base_dir = os.path.abspath(os.path.dirname(__file__))
    template_dir = os.path.join(base_dir, 'templates')
    static_dir = os.path.join(base_dir, 'static')
    
    app = Flask(__name__, 
                template_folder=template_dir, 
                static_folder=static_dir)
                
    # Debug: Print configured paths
    print(f"DEBUG: App root path: {app.root_path}")
    print(f"DEBUG: Template folder: {app.template_folder}")
    print(f"DEBUG: Static folder: {app.static_folder}")
    if os.path.exists(template_dir):
        print(f"DEBUG: Template dir exists: {template_dir}")
        print(f"DEBUG: Templates content: {os.listdir(template_dir)}")
    else:
        print(f"DEBUG: Template dir MISSING: {template_dir}")
    
    # Configure Logging
    if not os.path.exists('logs'):
        os.mkdir('logs')
    file_handler = RotatingFileHandler('logs/app.log', maxBytes=10240, backupCount=10)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
    app.logger.info('Moments startup')
    
    # Apply Config
    app.config['UPLOAD_FOLDER'] = config.UPLOAD_FOLDER
    app.config['MAX_CONTENT_LENGTH'] = config.MAX_CONTENT_LENGTH
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = config.SEND_FILE_MAX_AGE_DEFAULT
    app.config['TEMPLATES_AUTO_RELOAD'] = config.TEMPLATES_AUTO_RELOAD
    app.config['JSON_AS_ASCII'] = config.JSON_AS_ASCII
    
    # Ensure upload directory exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Register Blueprints
    from app.routes.main import main_bp
    from app.routes.auth import auth_bp
    from app.routes.diary import diary_bp
    from app.routes.api import api_bp
    
    app.register_blueprint(main_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(diary_bp)
    app.register_blueprint(api_bp)
    
    # Error Handlers
    from flask import redirect, url_for
    @app.errorhandler(413)
    def request_entity_too_large(e):
        app.logger.warning('Request entity too large')
        return redirect(url_for('main.camera'))
    
    @app.errorhandler(500)
    def internal_error(e):
        app.logger.error(f'Server Error: {e}')
        return "Internal Server Error", 500
        
    # One-time migration
    data_service.migrate_profile_store_once()

    # Context Processor for Global Variables
    @app.context_processor
    def inject_globals():
        return {
            'profile': data_service.normalize_profile_numbers(data_service.load_profile()),
            'now': datetime.now()
        }
    
    return app
