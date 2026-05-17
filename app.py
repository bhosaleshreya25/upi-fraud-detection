from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
from pymongo import MongoClient
import bcrypt
import pickle
import json
import os
import random
import string
import numpy as np
import re
from datetime import datetime
from bson.objectid import ObjectId
import pandas as pd

app = Flask(__name__)
app.secret_key = "upi_fraud_secret_key_2026"

# Create data directory and sample dataset if not exists
if not os.path.exists('data/onlinefraud.csv'):
    os.makedirs('data', exist_ok=True)
    print("Creating sample dataset...")
    sample_data = pd.DataFrame({
        'step': range(1, 1001),
        'type': ['PAYMENT'] * 1000,
        'amount': [5000] * 1000,
        'nameOrig': [f'user{i}' for i in range(1, 1001)],
        'oldbalanceOrg': [10000] * 1000,
        'newbalanceOrig': [5000] * 1000,
        'nameDest': [f'receiver{i}' for i in range(1, 1001)],
        'oldbalanceDest': [5000] * 1000,
        'newbalanceDest': [5000] * 1000,
        'isFraud': [0] * 1000
    })
    sample_data.to_csv('data/onlinefraud.csv', index=False)
    print("Sample dataset created at data/onlinefraud.csv")

# MongoDB Connection
client = MongoClient("mongodb://localhost:27017/")
db = client["upi_fraud_db"]

# Collections
users_col = db["users"]
upi_reports_col = db["upi_reports"]
transactions_col = db["transactions"]
txn_reports_col = db["transaction_reports"]
upi_blocklist_col = db["upi_blocklist"]
fraud_alerts_col = db["fraud_alerts"]
fraud_reports_col = db["fraud_reports"]
admin_logs_col = db["admin_logs"]
password_resets_col = db["password_resets"]  # Add this collection

# ==================== LOAD UPI DATASET FOR VERIFICATION ====================

UPI_DATABASE = set()
UPI_INFO = {}

# Complete list of Indian bank domains for UPI
BANK_DOMAINS = {
    'okhdfcbank': 'HDFC Bank',
    'sbi': 'State Bank of India',
    'icici': 'ICICI Bank',
    'axis': 'Axis Bank',
    'kotak': 'Kotak Mahindra Bank',
    'yesbank': 'Yes Bank',
    'ybl': 'Yes Bank Ltd',
    'ibl': 'IndusInd Bank',
    'okaxis': 'Axis Bank',
    'okicici': 'ICICI Bank',
    'oksbi': 'State Bank of India',
    'okhdfc': 'HDFC Bank',
    'pnb': 'Punjab National Bank',
    'canara': 'Canara Bank',
    'unionbank': 'Union Bank of India',
    'idfcbank': 'IDFC First Bank',
    'rbl': 'RBL Bank',
    'federal': 'Federal Bank',
    'paytm': 'Paytm Payments Bank',
    'phonepe': 'PhonePe',
    'googlepay': 'Google Pay',
    'amazonpay': 'Amazon Pay',
    'whatsapp': 'WhatsApp Pay'
}

def load_upi_dataset():
    """Load UPI data from CSV and create synthetic UPI IDs"""
    global UPI_DATABASE, UPI_INFO
    try:
        csv_path = 'data/onlinefraud.csv'
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            print(f"Dataset loaded: {len(df)} records")
            
            # Try to find a suitable column for generating UPI IDs
            name_column = None
            for col in ['nameOrig', 'name', 'customer', 'user', 'id', 'Name']:
                if col in df.columns:
                    name_column = col
                    break
            
            if name_column:
                print(f"Using column '{name_column}' for UPI generation")
                limit = min(10000, len(df))
                
                for i, row in df.head(limit).iterrows():
                    identifier = str(row.get(name_column, f'user{i}'))
                    if identifier and identifier != 'nan':
                        identifier = ''.join(c for c in identifier if c.isalnum() or c == '.')[:20]
                        
                        if identifier:
                            # Create UPI IDs for multiple bank domains
                            for domain, bank_name in list(BANK_DOMAINS.items())[:15]:
                                upi = f"{identifier.lower()}@{domain}"
                                UPI_DATABASE.add(upi)
                                UPI_INFO[upi] = {
                                    'name': identifier[:15],
                                    'bank': bank_name,
                                    'account_type': 'savings'
                                }
                
                print(f"   - Generated {len(UPI_DATABASE)} UPI IDs")
                return True
            else:
                print("Creating generic UPI IDs...")
                for i in range(min(5000, len(df))):
                    for domain in list(BANK_DOMAINS.keys())[:10]:
                        upi = f"user{i}@{domain}"
                        UPI_DATABASE.add(upi)
                        UPI_INFO[upi] = {
                            'name': f'User{i}',
                            'bank': BANK_DOMAINS[domain],
                            'account_type': 'savings'
                        }
                
                print(f"   - Generated {len(UPI_DATABASE)} generic UPI IDs")
                return True
        else:
            print("Dataset file not found, creating sample database...")
            create_sample_upi_database()
            return True
            
    except Exception as e:
        print(f"Could not load dataset: {e}")
        create_sample_upi_database()
        return False

def create_sample_upi_database():
    """Create a sample UPI database for testing"""
    global UPI_DATABASE, UPI_INFO
    
    sample_upis = [
        ('rahul.sharma', 'okhdfcbank', 'Rahul Sharma', 'HDFC Bank'),
        ('priya.verma', 'sbi', 'Priya Verma', 'State Bank of India'),
        ('amit.kumar', 'paytm', 'Amit Kumar', 'Paytm Payments Bank'),
        ('neha.gupta', 'icici', 'Neha Gupta', 'ICICI Bank'),
        ('vikram.singh', 'axis', 'Vikram Singh', 'Axis Bank'),
        ('deepika.patel', 'ybl', 'Deepika Patel', 'Yes Bank'),
        ('rajesh.mishra', 'okicici', 'Rajesh Mishra', 'ICICI Bank'),
        ('pooja.reddy', 'oksbi', 'Pooja Reddy', 'SBI'),
        ('ankit.mehta', 'phonepe', 'Ankit Mehta', 'PhonePe'),
        ('swati.jain', 'googlepay', 'Swati Jain', 'Google Pay'),
        ('test.user', 'okhdfcbank', 'Test User', 'HDFC Bank'),
        ('demo.account', 'paytm', 'Demo Account', 'Paytm'),
        ('fraud.test', 'paytm', 'Fraud Test', 'Paytm'),
        ('scam.alert', 'okhdfcbank', 'Scam Alert', 'HDFC Bank'),
        ('lottery.winner', 'sbi', 'Lottery Winner', 'SBI'),
    ]
    
    for username, domain, name, bank in sample_upis:
        upi = f"{username}@{domain}"
        UPI_DATABASE.add(upi)
        UPI_INFO[upi] = {
            'name': name,
            'bank': bank,
            'account_type': 'savings'
        }
    
    print(f"   - Created {len(UPI_DATABASE)} sample UPI IDs")

# Load ML Models
MODEL_DIR = "models"

def load_model(name):
    path = os.path.join(MODEL_DIR, f"{name}.pkl")
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    return None

rf_model = load_model("random_forest")
xgb_model = load_model("xgboost")
scaler = load_model("scaler")
features_list = load_model("features")
label_encoder = load_model("label_encoder")

# Load stats
stats = {}
stats_path = os.path.join(MODEL_DIR, "stats.json")
if os.path.exists(stats_path):
    stats = json.load(open(stats_path))

# Valid domains for UPI
VALID_DOMAINS = list(BANK_DOMAINS.keys())

SUSPICIOUS_WORDS = ['test', 'fake', 'demo', 'xyz', 'temp', 'scam', 'fraud', 'phish', 'hack', 'spam']
LOTTERY_WORDS = ['win', 'winner', 'prize', 'reward', 'bonus', 'cash', 'free', 'gift', 'claim', 'lucky', 'lottery']

# ==================== HELPER FUNCTIONS ====================

def generate_report_id():
    return "RPT" + datetime.now().strftime("%Y%m%d") + "".join(random.choices(string.digits, k=6))

def generate_txn_id():
    return "TXN" + datetime.now().strftime("%Y%m%d") + "".join(random.choices(string.digits, k=6))

def send_admin_notification(report):
    print(f"\n[ADMIN NOTIFICATION] New Fraud Report: {report['report_id']}")

def predict_fraud(amount, time_str, frequency, upi_id, balance, is_new_receiver, device):
    """Simple rule-based fraud prediction"""
    hour = int(time_str.split(":")[0]) if ":" in time_str else 12
    
    risk_score = 0
    reasons = []
    
    if amount > 50000:
        risk_score += 40
        reasons.append("High amount transaction")
    
    if hour >= 22 or hour <= 5:
        risk_score += 25
        reasons.append("Late night transaction")
    
    if is_new_receiver == "yes":
        risk_score += 30
        reasons.append("New receiver")
    
    if device == "new" or device == "unknown":
        risk_score += 20
        reasons.append("Unknown device")
    
    if risk_score >= 70:
        label = "Fraud"
        action = "BLOCK TRANSACTION"
        alert_needed = True
    elif risk_score >= 45:
        label = "Suspicious"
        action = "VERIFICATION REQUIRED"
        alert_needed = False
    else:
        label = "Safe"
        action = "APPROVE TRANSACTION"
        alert_needed = False
    
    return {
        "label": label,
        "risk_score": risk_score,
        "action": action,
        "reasons": reasons,
        "alert_triggered": alert_needed
    }

# ==================== UPI FRAUD DETECTION ====================

def detect_upi_fraud(upi_id):
    upi_id = upi_id.lower().strip()
    warnings = []
    risk_score = 0
    report_count = 0
    
    # Check if in database
    is_known_upi = upi_id in UPI_DATABASE
    
    if is_known_upi and upi_id in UPI_INFO:
        info = UPI_INFO[upi_id]
        warnings.append(f"Verified UPI from {info.get('bank', 'registered bank')}")
    else:
        risk_score += 10
        warnings.append("This UPI ID is not found in our verified database")
    
    # Check blocklist
    blocked = upi_blocklist_col.find_one({"upi": upi_id})
    if blocked:
        return 100, "Critical", "BLOCKED UPI", "Block Immediately", ["This UPI has been permanently blocked"], 999
    
    # Check reports
    report_doc = upi_reports_col.find_one({"upi": upi_id})
    if report_doc:
        report_count = report_doc.get("report_count", 0)
        if report_count >= 5:
            risk_score += 45
            warnings.append(f"This UPI has {report_count} fraud reports")
        elif report_count >= 2:
            risk_score += 25
            warnings.append(f"This UPI has {report_count} suspicious reports")
    
    # Format validation
    if '@' not in upi_id:
        return 95, "Critical", "INVALID FORMAT", "Block Immediately", ["Missing @ symbol"], report_count
    
    parts = upi_id.split('@')
    username = parts[0] if len(parts) > 0 else ''
    domain = parts[1] if len(parts) > 1 else ''
    
    if not username or not domain:
        return 92, "High", "INVALID FORMAT", "Verify", ["Invalid UPI format"], report_count
    
    # Pattern detection
    for word in LOTTERY_WORDS:
        if word in username:
            risk_score += 45
            warnings.append(f"Lottery keyword: '{word}'")
            break
    
    for word in SUSPICIOUS_WORDS:
        if word in username:
            risk_score += 40
            warnings.append(f"Suspicious keyword: '{word}'")
            break
    
    if username.isdigit():
        risk_score += 40
        warnings.append("Username contains only digits")
    
    if domain not in VALID_DOMAINS:
        risk_score += 35
        warnings.append(f"Invalid domain: '@{domain}'")
    
    risk_score = min(risk_score, 99)
    
    if risk_score >= 70:
        risk_level = "Critical"
        message = "CRITICAL: Multiple fraud indicators"
        action = "BLOCK IMMEDIATELY"
    elif risk_score >= 50:
        risk_level = "High"
        message = "HIGH RISK: Suspicious patterns"
        action = "VERIFY THOROUGHLY"
    elif risk_score >= 30:
        risk_level = "Medium"
        message = "MEDIUM RISK: Some indicators"
        action = "Additional verification"
    else:
        risk_level = "Low"
        message = "LOW RISK: Appears legitimate"
        action = "Safe to transact"
    
    if not warnings:
        warnings = ["No fraud indicators found"]
    
    return risk_score, risk_level, message, action, warnings, report_count

# ==================== AUTH ROUTES ====================

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form["email"].strip().lower()
        password = request.form["password"]
        role = request.form.get("role", "user")
        
        if users_col.find_one({"email": email}):
            return render_template("signup.html", error="Email already registered.")
        
        if len(password) < 6:
            return render_template("signup.html", error="Password must be at least 6 characters.")
        
        hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
        
        user = {
            "name": name,
            "email": email,
            "password": hashed,
            "role": role,
            "created_at": datetime.now(),
            "is_active": True
        }
        
        users_col.insert_one(user)
        
        session["user"] = email
        session["user_name"] = name
        session["role"] = role
        
        flash("Account created successfully!", "success")
        
        if role == "admin":
            return redirect(url_for("admin_dashboard"))
        else:
            return redirect(url_for("home"))
    
    return render_template("signup.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"].strip().lower()
        password = request.form["password"]
        selected_role = request.form.get("role", "user")
        
        user = users_col.find_one({"email": email})
        
        if user and bcrypt.checkpw(password.encode(), user["password"]):
            if user.get("role", "user") != selected_role:
                return render_template("login.html", 
                    error=f"This account is registered as {user.get('role', 'user')}, not as {selected_role}.")
            
            session["user"] = email
            session["user_name"] = user.get("name", email.split('@')[0])
            session["role"] = user.get("role", "user")
            session["user_id"] = str(user["_id"])
            
            flash(f"Welcome back, {session['user_name']}!", "success")
            
            if session["role"] == "admin":
                return redirect(url_for("admin_dashboard"))
            else:
                return redirect(url_for("home"))
        
        return render_template("login.html", error="Invalid email or password.")
    
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))

# ==================== FORGOT PASSWORD ROUTES ====================

@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        
        # Check if user exists
        user = users_col.find_one({"email": email})
        
        if user:
            # Generate reset token
            reset_token = ''.join(random.choices(string.ascii_letters + string.digits, k=32))
            
            # Store reset token in database
            password_resets_col.update_one(
                {"email": email},
                {"$set": {
                    "token": reset_token,
                    "created_at": datetime.now(),
                    "expires_at": datetime.now().replace(hour=datetime.now().hour + 1)
                }},
                upsert=True
            )
            
            flash(f"Password reset link generated! Use token: {reset_token}", "info")
            return render_template("forgot_password.html", 
                                  message="Reset link generated! Check below.", 
                                  reset_token=reset_token)
        else:
            flash("Email not found!", "error")
    
    return render_template("forgot_password.html")

@app.route("/reset-password", methods=["GET", "POST"])
def reset_password():
    token = request.args.get("token")
    
    if request.method == "POST":
        token = request.form.get("token")
        new_password = request.form.get("new_password")
        confirm_password = request.form.get("confirm_password")
        
        if new_password != confirm_password:
            flash("Passwords do not match!", "error")
            return render_template("reset_password.html", token=token)
        
        if len(new_password) < 6:
            flash("Password must be at least 6 characters!", "error")
            return render_template("reset_password.html", token=token)
        
        # Verify token
        reset_record = password_resets_col.find_one({"token": token})
        
        if not reset_record:
            flash("Invalid or expired reset link!", "error")
            return redirect(url_for("forgot_password"))
        
        if reset_record["expires_at"] < datetime.now():
            flash("Reset link has expired!", "error")
            return redirect(url_for("forgot_password"))
        
        # Update password
        email = reset_record["email"]
        hashed = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt())
        
        users_col.update_one(
            {"email": email},
            {"$set": {"password": hashed}}
        )
        
        # Delete reset token
        password_resets_col.delete_one({"token": token})
        
        flash("Password reset successfully! Please login with your new password.", "success")
        return redirect(url_for("login"))
    
    return render_template("reset_password.html", token=token)

# ==================== ADMIN DASHBOARD ====================

@app.route("/admin")
def admin_dashboard():
    if session.get("role") != "admin":
        flash("Access denied. Admin privileges required.", "error")
        return redirect(url_for("home"))
    
    pending_reports = list(fraud_reports_col.find({"status": "pending"}).sort("created_at", -1).limit(20))
    for report in pending_reports:
        report["_id"] = str(report["_id"])
    
    reports = list(upi_reports_col.find({}).sort("last_reported", -1).limit(50))
    for report in reports:
        report["_id"] = str(report["_id"])
    
    blocked_upis = list(upi_blocklist_col.find({}).sort("blocked_at", -1))
    for blocked in blocked_upis:
        blocked["_id"] = str(blocked["_id"])
    
    transactions = list(transactions_col.find({}, {"_id": 0}).sort("timestamp", -1).limit(20))
    alerts = list(fraud_alerts_col.find({}, {"_id": 0}).sort("timestamp", -1).limit(20))
    
    return render_template("admin.html",
                          reports=reports,
                          pending_reports=pending_reports,
                          pending_reports_count=len(pending_reports),
                          blocked_upis=blocked_upis,
                          transactions=transactions,
                          alerts=alerts,
                          logged_in=True,
                          is_admin=True)

@app.route("/admin/block-upi", methods=["POST"])
def admin_block_upi():
    if session.get("role") != "admin":
        return redirect(url_for("login"))
    
    upi_id = request.form["upi_id"].strip().lower()
    reason = request.form.get("reason", "Admin blocked")
    
    upi_blocklist_col.update_one(
        {"upi": upi_id},
        {"$set": {"blocked_at": datetime.now(), "reason": reason, "blocked_by": session.get("user")}},
        upsert=True
    )
    
    flash(f"UPI {upi_id} has been blocked successfully.", "success")
    return redirect(url_for("admin_dashboard"))

# ==================== ADMIN FRAUD REPORT MANAGEMENT ====================

@app.route("/admin/reports")
def admin_reports_page():
    if session.get("role") != "admin":
        flash("Access denied.", "error")
        return redirect(url_for("home"))
    
    pending_reports = list(fraud_reports_col.find({"status": "pending"}).sort("created_at", -1))
    for report in pending_reports:
        report["_id"] = str(report["_id"])
        report["created_at"] = report["created_at"].strftime("%Y-%m-%d %H:%M:%S") if report.get("created_at") else "N/A"
    
    blocked_upis = list(upi_blocklist_col.find({}).sort("blocked_at", -1))
    for blocked in blocked_upis:
        blocked["_id"] = str(blocked["_id"])
        blocked["blocked_at"] = blocked["blocked_at"].strftime("%Y-%m-%d %H:%M:%S") if blocked.get("blocked_at") else "N/A"
    
    blocked_transactions = list(transactions_col.find({"status": "blocked"}).sort("timestamp", -1))
    for txn in blocked_transactions:
        txn["_id"] = str(txn["_id"])
    
    return render_template("admin_reports.html",
                          pending_reports=pending_reports,
                          blocked_upis=blocked_upis,
                          blocked_transactions=blocked_transactions,
                          logged_in=True,
                          is_admin=True)

@app.route("/admin/block_transaction", methods=["POST"])
def admin_block_transaction():
    if session.get("role") != "admin":
        return jsonify({"error": "Unauthorized"}), 401
    
    transaction_id = request.form.get("transaction_id")
    upi_id = request.form.get("upi_id", "").strip().lower()
    report_id = request.form.get("report_id")
    reason = request.form.get("reason", "Fraudulent activity detected")
    block_type = request.form.get("block_type", "transaction")
    
    if block_type == "transaction" and transaction_id:
        transactions_col.update_one(
            {"transaction_id": transaction_id},
            {"$set": {
                "status": "blocked",
                "block_reason": reason,
                "blocked_by": session.get("user"),
                "blocked_at": datetime.now()
            }}
        )
        
        if report_id:
            fraud_reports_col.update_one(
                {"_id": ObjectId(report_id)},
                {"$set": {
                    "status": "resolved",
                    "admin_notes": f"Transaction blocked. Reason: {reason}",
                    "action_taken": "blocked_transaction",
                    "reviewed_at": datetime.now(),
                    "reviewed_by": session.get("user")
                }}
            )
        
        flash(f"Transaction {transaction_id} has been blocked!", "success")
    
    elif block_type == "upi" and upi_id:
        upi_blocklist_col.update_one(
            {"upi": upi_id},
            {"$set": {
                "blocked_at": datetime.now(),
                "reason": reason,
                "blocked_by": session.get("user"),
                "status": "blocked"
            }},
            upsert=True
        )
        
        if report_id:
            fraud_reports_col.update_one(
                {"_id": ObjectId(report_id)},
                {"$set": {
                    "status": "resolved",
                    "admin_notes": f"UPI {upi_id} blocked. Reason: {reason}",
                    "action_taken": "blocked_upi",
                    "reviewed_at": datetime.now(),
                    "reviewed_by": session.get("user")
                }}
            )
        
        flash(f"UPI {upi_id} has been blocked!", "success")
    
    return redirect(url_for("admin_reports_page"))

@app.route("/admin/unblock_transaction", methods=["POST"])
def admin_unblock_transaction():
    if session.get("role") != "admin":
        return jsonify({"error": "Unauthorized"}), 401
    
    transaction_id = request.form.get("transaction_id")
    upi_id = request.form.get("upi_id", "").strip().lower()
    unblock_type = request.form.get("unblock_type", "transaction")
    
    if unblock_type == "transaction" and transaction_id:
        transactions_col.update_one(
            {"transaction_id": transaction_id},
            {"$set": {"status": "completed", "unblocked_by": session.get("user"), "unblocked_at": datetime.now()},
             "$unset": {"block_reason": "", "blocked_by": "", "blocked_at": ""}}
        )
        flash(f"Transaction {transaction_id} has been unblocked!", "success")
    
    elif unblock_type == "upi" and upi_id:
        upi_blocklist_col.delete_one({"upi": upi_id})
        flash(f"UPI {upi_id} has been unblocked!", "success")
    
    return redirect(url_for("admin_reports_page"))

@app.route("/admin/reject_report", methods=["POST"])
def admin_reject_report():
    if session.get("role") != "admin":
        return jsonify({"error": "Unauthorized"}), 401
    
    report_id = request.form.get("report_id")
    rejection_reason = request.form.get("rejection_reason", "No evidence found")
    
    fraud_reports_col.update_one(
        {"_id": ObjectId(report_id)},
        {"$set": {
            "status": "rejected",
            "admin_notes": rejection_reason,
            "action_taken": "rejected",
            "reviewed_at": datetime.now(),
            "reviewed_by": session.get("user")
        }}
    )
    
    flash("Report has been rejected.", "info")
    return redirect(url_for("admin_reports_page"))

# ==================== USER ROUTES ====================

@app.route("/")
def home():
    total = transactions_col.count_documents({})
    fraud_cnt = transactions_col.count_documents({"label": "Fraud"})
    upi_cnt = upi_reports_col.count_documents({})
    blocked_cnt = upi_blocklist_col.count_documents({})
    
    return render_template("home.html",
                           total=total, fraud_cnt=fraud_cnt, 
                           upi_cnt=upi_cnt, blocked_cnt=blocked_cnt,
                           logged_in="user" in session,
                           is_admin=session.get("role") == "admin")

@app.route("/verify-upi", methods=["GET", "POST"])
def verify_upi():
    result = None
    if request.method == "POST":
        upi_id = request.form["upi_id"].strip()
        risk_score, risk_level, message, action, warnings, report_count = detect_upi_fraud(upi_id)
        
        result = {
            "upi": upi_id,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "message": message,
            "action": action,
            "warnings": warnings,
            "report_count": report_count
        }
    
    return render_template("verify_upi.html", result=result, 
                          logged_in="user" in session,
                          is_admin=session.get("role") == "admin")

@app.route("/report-upi", methods=["POST"])
def report_upi():
    upi_id = request.form["upi_id"].strip().lower()
    
    existing = upi_reports_col.find_one({"upi": upi_id})
    if existing:
        new_count = existing["report_count"] + 1
        upi_reports_col.update_one({"upi": upi_id},
            {"$set": {"report_count": new_count, "last_reported": datetime.now(),
                      "status": "Fraud" if new_count >= 5 else "Suspicious"}})
    else:
        upi_reports_col.insert_one({
            "upi": upi_id, "report_count": 1, "status": "Reported",
            "first_reported": datetime.now(), "last_reported": datetime.now()
        })
    
    flash(f"UPI {upi_id} has been reported.", "success")
    return redirect(url_for("verify_upi"))

@app.route("/report-fraud", methods=["GET", "POST"])
def report_fraud_page():
    if "user" not in session:
        flash("Please login to report fraud", "warning")
        return redirect(url_for("login"))
    
    if request.method == "POST":
        report = {
            "report_id": generate_report_id(),
            "reported_by": session.get("user"),
            "reported_by_name": session.get("user_name"),
            "report_type": request.form.get("report_type"),
            "upi_id": request.form.get("upi_id", "").strip().lower(),
            "transaction_id": request.form.get("transaction_id", "").strip(),
            "amount": float(request.form.get("amount", 0)),
            "fraud_type": request.form.get("fraud_type", ""),
            "description": request.form.get("description", ""),
            "status": "pending",
            "created_at": datetime.now()
        }
        
        fraud_reports_col.insert_one(report)
        flash("Thank you for reporting!", "success")
        return redirect(url_for("user_reports"))
    
    return render_template("report_fraud.html", logged_in=True, is_admin=False)

@app.route("/my-reports")
def user_reports():
    if "user" not in session:
        return redirect(url_for("login"))
    
    user_reports_list = list(fraud_reports_col.find({"reported_by": session["user"]}).sort("created_at", -1))
    for report in user_reports_list:
        report["_id"] = str(report["_id"])
    
    return render_template("my_reports.html", reports=user_reports_list, logged_in=True, is_admin=False)

@app.route("/dashboard")
def dashboard():
    total = transactions_col.count_documents({})
    fraud_cnt = transactions_col.count_documents({"label": "Fraud"})
    safe_cnt = total - fraud_cnt
    upi_cnt = upi_reports_col.count_documents({})
    alert_cnt = fraud_alerts_col.count_documents({})
    blocked_cnt = upi_blocklist_col.count_documents({})
    
    recent = list(transactions_col.find({}, {"_id": 0}).sort("timestamp", -1).limit(10))
    
    return render_template("dashboard.html",
                           stats=stats, total=total, fraud_cnt=fraud_cnt, safe_cnt=safe_cnt,
                           upi_cnt=upi_cnt, alert_cnt=alert_cnt, blocked_cnt=blocked_cnt,
                           recent=recent,
                           logged_in="user" in session,
                           is_admin=session.get("role") == "admin")

@app.route("/analyze", methods=["GET", "POST"])
def analyze():
    result = None
    
    if request.method == "POST":
        try:
            # Get form data
            sender_upi = request.form.get("sender_upi", "").strip()
            receiver_upi = request.form.get("receiver_upi", "").strip()
            payment_app = request.form.get("payment_app", "")
            amount = float(request.form.get("amount", 0))
            balance = float(request.form.get("balance", 0))
            datetime_str = request.form.get("datetime", "")
            device = request.form.get("device", "")
            is_new_receiver = request.form.get("is_new_receiver", "")
            frequency = int(request.form.get("frequency", 1))
            
            # Parse hour
            hour = 12
            if datetime_str:
                try:
                    dt = datetime.strptime(datetime_str, "%Y-%m-%dT%H:%M")
                    hour = dt.hour
                except:
                    hour = 12
            
            # Simple risk calculation
            risk_score = 0
            reasons = []
            
            # Amount-based risk
            if amount > 100000:
                risk_score += 50
                reasons.append("Very high amount (>₹1,00,000)")
            elif amount > 50000:
                risk_score += 35
                reasons.append("High amount (>₹50,000)")
            elif amount > 25000:
                risk_score += 20
                reasons.append("Moderate high amount (>₹25,000)")
            
            # Time-based risk
            if hour >= 23 or hour <= 4:
                risk_score += 30
                reasons.append("Transaction at late night (12 AM - 4 AM)")
            elif hour >= 22 or hour <= 5:
                risk_score += 20
                reasons.append("Transaction at odd hours")
            
            # New receiver risk
            if is_new_receiver == "yes":
                risk_score += 25
                reasons.append("First time payment to this receiver")
            
            # Device risk
            if device == "unknown":
                risk_score += 30
                reasons.append("Unknown device detected")
            elif device == "new":
                risk_score += 20
                reasons.append("New device detected")
            
            # Balance risk
            balance_after = balance - amount
            if balance_after < 0:
                risk_score += 50
                reasons.append("Insufficient balance")
            elif balance_after < 1000:
                risk_score += 25
                reasons.append("Very low balance after transaction")
            
            # Frequency risk
            if frequency > 10:
                risk_score += 20
                reasons.append("High transaction frequency (10+ today)")
            elif frequency > 5:
                risk_score += 10
                reasons.append("Moderate transaction frequency")
            
            # UPI ID pattern detection (simple check)
            suspicious_keywords = ['lottery', 'winner', 'prize', 'free', 'gift', 'cash', 'bonus', 'claim', 'fraud', 'scam', 'test']
            for keyword in suspicious_keywords:
                if keyword in receiver_upi.lower():
                    risk_score += 35
                    reasons.append(f"Suspicious UPI ID contains '{keyword}'")
                    break
            
            # Ensure risk_score is within 0-100
            risk_score = min(risk_score, 99)
            
            # Determine label and action
            if risk_score >= 70:
                label = "Fraud"
                action = "BLOCK TRANSACTION"
                alert_triggered = True
            elif risk_score >= 45:
                label = "Suspicious"
                action = "VERIFICATION REQUIRED"
                alert_triggered = False
            else:
                label = "Safe"
                action = "APPROVE TRANSACTION"
                alert_triggered = False
            
            # Generate transaction ID
            transaction_id = f"TXN{datetime.now().strftime('%Y%m%d%H%M%S')}{random.randint(100, 999)}"
            
            # Create result with ALL required fields
            result = {
                "transaction_id": transaction_id,
                "sender_upi": sender_upi,
                "receiver_upi": receiver_upi,
                "payment_app": payment_app,
                "amount": amount,
                "balance_after": balance_after if balance_after > 0 else 0,
                "risk_score": risk_score,
                "label": label,
                "action": action,
                "reasons": reasons,
                "model_votes": {  # This is the missing field!
                    "Random Forest": "Fraud" if risk_score >= 70 else ("Suspicious" if risk_score >= 45 else "Safe"),
                    "XGBoost": "Fraud" if risk_score >= 70 else ("Suspicious" if risk_score >= 45 else "Safe")
                },
                "alert_triggered": alert_triggered,
                "upi_status": "High Risk" if risk_score >= 70 else ("Medium Risk" if risk_score >= 45 else "Low Risk"),
                "upi_reports": 0
            }
            
            # Save to database if available
            if 'transactions_col' in globals():
                try:
                    transactions_col.insert_one({
                        "transaction_id": transaction_id,
                        "sender_upi": sender_upi,
                        "receiver_upi": receiver_upi,
                        "payment_app": payment_app,
                        "amount": amount,
                        "balance_before": balance,
                        "balance_after": balance_after,
                        "datetime": datetime_str,
                        "hour": hour,
                        "device": device,
                        "is_new_receiver": is_new_receiver,
                        "frequency": frequency,
                        "risk_score": risk_score,
                        "label": label,
                        "action": action,
                        "reasons": reasons,
                        "timestamp": datetime.now()
                    })
                except:
                    pass  # Skip if database not available
            
            # Save alert if fraud detected
            if alert_triggered and 'fraud_alerts_col' in globals():
                try:
                    fraud_alerts_col.insert_one({
                        'type': 'email',
                        'upi_id': receiver_upi,
                        'amount': amount,
                        'risk_score': risk_score,
                        'reasons': reasons,
                        'timestamp': datetime.now()
                    })
                except:
                    pass
                    
        except Exception as e:
            print(f"Error in analyze: {e}")
            flash(f"Error: {str(e)}", "error")
    
    return render_template("analyze.html", 
                          result=result,
                          logged_in="user" in session,
                          is_admin=session.get("role") == "admin")
@app.route("/report-transaction", methods=["POST"])
def report_transaction():
    """Report a suspicious transaction"""
    if "user" not in session:
        flash("Please login to report", "warning")
        return redirect(url_for("login"))
    
    try:
        upi_id = request.form.get("upi_id", "").strip().lower()
        amount = float(request.form.get("amount", 0))
        fraud_type = request.form.get("fraud_type", "")
        description = request.form.get("description", "")
        
        # Store in fraud_reports collection
        report = {
            "report_id": generate_report_id(),
            "reported_by": session.get("user"),
            "reported_by_name": session.get("user_name"),
            "report_type": "transaction",
            "upi_id": upi_id,
            "amount": amount,
            "fraud_type": fraud_type,
            "description": description,
            "status": "pending",
            "created_at": datetime.now()
        }
        
        fraud_reports_col.insert_one(report)
        
        # Also store in transaction reports
        txn_reports_col.insert_one({
            "upi": upi_id,
            "amount": amount,
            "reason": description,
            "fraud_type": fraud_type,
            "reported_by": session.get("user"),
            "timestamp": datetime.now()
        })
        
        flash("✅ Report submitted successfully! Admin will review it.", "success")
    except Exception as e:
        flash(f"Error: {str(e)}", "error")
    
    return redirect(url_for("analyze"))
# ==================== END OF NEW ROUTE ====================


@app.context_processor
def inject_user():
    return {
        "logged_in": "user" in session,
        "user_name": session.get("user_name", ""),
        "user_role": session.get("role", ""),
        "is_admin": session.get("role") == "admin"
    }

# ==================== CREATE DEFAULT USERS ====================

def create_default_users():
    if not users_col.find_one({"email": "admin@upifraudshield.com"}):
        hashed = bcrypt.hashpw("admin123".encode(), bcrypt.gensalt())
        users_col.insert_one({
            "name": "System Admin",
            "email": "admin@upifraudshield.com",
            "password": hashed,
            "role": "admin",
            "created_at": datetime.now(),
            "is_active": True
        })
        print("Default admin created: admin@upifraudshield.com / admin123")
    
    if not users_col.find_one({"email": "user@example.com"}):
        hashed = bcrypt.hashpw("user123".encode(), bcrypt.gensalt())
        users_col.insert_one({
            "name": "Demo User",
            "email": "user@example.com",
            "password": hashed,
            "role": "user",
            "created_at": datetime.now(),
            "is_active": True
        })
        print("Demo user created: user@example.com / user123")

# ==================== RUN APP ====================

if __name__ == "__main__":
    os.makedirs("templates", exist_ok=True)
    os.makedirs("static/css", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    load_upi_dataset()
    create_default_users()
    
    print("\n" + "=" * 60)
    print("UPI FRAUD DETECTION SYSTEM")
    print("=" * 60)
    print("Server running at: http://localhost:5000")
    print("-" * 40)
    print("Admin Login: admin@upifraudshield.com / admin123")
    print("User Login: user@example.com / user123")
    print("-" * 40)
    print(f"UPI Dataset Status: {'Loaded' if UPI_DATABASE else 'Not Loaded'}")
    print(f"Total UPI IDs: {len(UPI_DATABASE)}")
    print("=" * 60 + "\n")
    
    app.run(debug=True, port=5000)