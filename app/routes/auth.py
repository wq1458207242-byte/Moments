from flask import Blueprint, request, jsonify
from datetime import datetime
import uuid
from app.services.data_service import data_service
from app.utils.helpers import _hash, _create_token

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/auth/register', methods=['POST'])
def register():
    data = request.json or {}
    identifier = (data.get("email") or data.get("phone") or "").strip().lower()
    password = str(data.get("password") or "")
    if not identifier or not password:
        return jsonify({"error": "missing credentials"}), 400
    users = data_service.load_users()
    for u in users:
        if u.get("identifier") == identifier:
            return jsonify({"error": "exists"}), 409
    user_id = uuid.uuid4().hex[:12]
    token = _create_token()
    users.append({
        "id": user_id,
        "identifier": identifier,
        "password_hash": _hash(password),
        "token": token,
        "created_at": datetime.utcnow().isoformat() + "Z",
    })
    data_service.save_users(users)
    return jsonify({"user_id": user_id, "token": token})

@auth_bp.route('/auth/login', methods=['POST'])
def login():
    data = request.json or {}
    identifier = (data.get("email") or data.get("phone") or "").strip().lower()
    password = str(data.get("password") or "")
    users = data_service.load_users()
    for u in users:
        if u.get("identifier") == identifier and u.get("password_hash") == _hash(password):
            u["token"] = _create_token()
            data_service.save_users(users)
            return jsonify({"user_id": u.get("id"), "token": u.get("token")})
    return jsonify({"error": "invalid"}), 401
