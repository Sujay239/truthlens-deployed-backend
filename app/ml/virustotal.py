import os
import requests
import hashlib
import time
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("VIRUSTOTAL_API_KEY")
BASE_URL = "https://www.virustotal.com/api/v3"

class VirusTotalClient:
    def __init__(self):
        if not API_KEY:
            raise ValueError("VIRUSTOTAL_API_KEY not found in environment variables")
        self.headers = {
            "x-apikey": API_KEY
        }

    def _get_hash(self, file_bytes):
        sha256 = hashlib.sha256()
        sha256.update(file_bytes)
        return sha256.hexdigest()

    def scan_file(self, file_bytes, filename):
        """
        Scans a file by checking its hash against VirusTotal.
        If hash is unknown, it (optionally) uploads the file.
        For this implementation, we prioritize hash checking to rely on VT's massive db.
        """
        file_hash = self._get_hash(file_bytes)
        print(f"[VirusTotal] Checking hash: {file_hash} for {filename}")
        
        # 1. Check Hash
        url = f"{BASE_URL}/files/{file_hash}"
        response = requests.get(url, headers=self.headers)
        
        if response.status_code == 200:
            data = response.json()
            return self._parse_report(data)
        elif response.status_code == 404:
            # File not known. In a full production app, we would upload here.
            # However, for responsiveness in this demo, we might return "Unknown".
            # Or we can support upload (limit 32MB).
            print(f"[VirusTotal] Hash not found. Uploading file...")
            return self._upload_and_scan(file_bytes, filename)
        else:
            print(f"[VirusTotal] Error: {response.status_code} - {response.text}")
            return {"error": f"VirusTotal Error: {response.status_code}"}

    def _upload_and_scan(self, file_bytes, filename):
        # Limit size to avoid issues (VT Free has limits, but we try)
        if len(file_bytes) > 32 * 1024 * 1024:
             return {"error": "File too large for upload (Limit 32MB)"}

        url = f"{BASE_URL}/files"
        files = {"file": (filename, file_bytes)}
        response = requests.post(url, headers=self.headers, files=files)
        
        if response.status_code == 200:
            data = response.json()
            analysis_id = data['data']['id']
            # Wait for analysis to complete
            return self._wait_for_analysis(analysis_id)
        else:
            return {"error": f"Upload failed: {response.status_code}"}

    def scan_url(self, target_url):
        print(f"[VirusTotal] Scanning URL: {target_url}")
        
        # 1. Submit URL
        url = f"{BASE_URL}/urls"
        payload = {"url": target_url}
        response = requests.post(url, headers=self.headers, data=payload)
        
        if response.status_code == 200:
            data = response.json()
            analysis_id = data['data']['id']
            return self._wait_for_analysis(analysis_id)

        else:
             return {"error": f"URL Scan failed: {response.status_code}"}

    def _wait_for_analysis(self, analysis_id):
        """Polls the analysis endpoint until completion or timeout."""
        report_url = f"{BASE_URL}/analyses/{analysis_id}"
        print(f"[VirusTotal] Waiting for analysis: {analysis_id}")
        
        # Wait up to 60 seconds (VT can be slow)
        max_retries = 30 
        for i in range(max_retries):
            response = requests.get(report_url, headers=self.headers)
            if response.status_code == 200:
                data = response.json()
                status = data['data']['attributes']['status']
                if status == 'completed':
                    return self._parse_analysis(data)
            
            time.sleep(2) # Poll every 2 seconds
            
        return {
            "label": "Queued", 
            "score": 0, 
            "threat_level": "Unknown", 
            "analysis": "Analysis timed out. Please check back later."
        }

    def _parse_report(self, data):
        """Parses /files/{id} response"""
        attr = data['data']['attributes']
        stats = attr['last_analysis_stats']
        
        malicious = stats['malicious']
        suspicious = stats['suspicious']
        harmless = stats['harmless']
        undetected = stats['undetected']
        
        total_engines = malicious + suspicious + harmless + undetected
        score = int((malicious / total_engines) * 100) if total_engines > 0 else 0
        
        if malicious > 0:
            label = "Malicious"
            threat_level = "High" if malicious > 5 else "Medium"
        elif suspicious > 0:
            label = "Suspicious"
            threat_level = "Low"
        else:
            label = "Clean"
            threat_level = "None"
            
        engines_flagged = [k for k,v in attr['last_analysis_results'].items() if v['category'] == 'malicious']
        top_threat = engines_flagged[0] if engines_flagged else "None"
            
        return {
            "label": label,
            "score": score, # Normalized 0-100 roughly
            "malicious_count": malicious,
            "total_engines": total_engines,
            "threat_level": threat_level,
            "signature": top_threat,
            "analysis": f"Flagged by {malicious}/{total_engines} vendors."
        }
        
    def _parse_analysis(self, data):
        """Parses /analyses/{id} response (for URL/Uploads)"""
        stats = data['data']['attributes']['stats']
        
        malicious = stats['malicious']
        suspicious = stats['suspicious']
        
        if malicious > 0:
            label = "Malicious"
            threat_level = "High"
        elif suspicious > 0:
            label = "Suspicious"
            threat_level = "Low"
        else:
            label = "Clean"
            threat_level = "None"
            
        return {
            "label": label,
            "score": malicious * 10, # Rough score
            "malicious_count": malicious,
            "threat_level": threat_level,
            "signature": "URL Threat",
            "analysis": f"Flagged by {malicious} vendors."
        }
