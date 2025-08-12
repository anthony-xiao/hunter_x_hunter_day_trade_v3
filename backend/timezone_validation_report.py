#!/usr/bin/env python3
"""
Comprehensive Timezone Validation Script
Validates timezone consistency across the entire trading system
"""

import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any
import subprocess

class TimezoneValidator:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.issues = []
        self.summary = {
            'total_files_scanned': 0,
            'files_with_issues': 0,
            'total_issues': 0,
            'critical_issues': 0,
            'warning_issues': 0
        }
    
    def validate_postgresql_timezone(self) -> Dict[str, Any]:
        """Validate PostgreSQL timezone configuration"""
        try:
            # Check database timezone
            result = subprocess.run(
                ['psql', '-d', 'algo_trading', '-c', 'SHOW timezone;', '-t'],
                capture_output=True, text=True
            )
            
            if result.returncode == 0:
                timezone_setting = result.stdout.strip()
                is_utc = timezone_setting.upper() == 'UTC'
                
                return {
                    'status': 'success',
                    'timezone': timezone_setting,
                    'is_utc': is_utc,
                    'issue': None if is_utc else f"Database timezone is '{timezone_setting}', should be 'UTC'"
                }
            else:
                return {
                    'status': 'error',
                    'error': result.stderr,
                    'issue': 'Could not connect to PostgreSQL database'
                }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'issue': f'PostgreSQL validation failed: {e}'
            }
    
    def scan_datetime_usage(self) -> List[Dict[str, Any]]:
        """Scan for problematic datetime usage patterns"""
        issues = []
        
        # Patterns to look for
        patterns = {
            'datetime_now_no_tz': r'datetime\.now\(\)',
            'datetime_utcnow': r'datetime\.utcnow\(\)',
            'time_time': r'time\.time\(\)',
            'pd_timestamp_now': r'pd\.Timestamp\.now\(\)',
            'datetime_fromtimestamp_no_tz': r'datetime\.fromtimestamp\([^,)]+\)(?!.*tz=)',
        }
        
        # Files to scan (exclude venv and other non-application directories)
        exclude_patterns = ['venv', '__pycache__', '.git', 'node_modules', '.pytest_cache']
        
        # Scan current directory (backend) and its subdirectories
        for file_path in self.project_root.rglob('*.py'):
            # Skip excluded directories
            if any(exclude in str(file_path) for exclude in exclude_patterns):
                continue
                
            self.summary['total_files_scanned'] += 1
            file_issues = self._scan_file(file_path, patterns)
            if file_issues:
                issues.extend(file_issues)
                self.summary['files_with_issues'] += 1
        
        return issues
    
    def _scan_file(self, file_path: Path, patterns: Dict[str, str]) -> List[Dict[str, Any]]:
        """Scan a single file for timezone issues"""
        issues = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
            
            for pattern_name, pattern in patterns.items():
                matches = re.finditer(pattern, content, re.MULTILINE)
                
                for match in matches:
                    # Find line number
                    line_num = content[:match.start()].count('\n') + 1
                    line_content = lines[line_num - 1].strip()
                    
                    # Determine severity
                    severity = self._determine_severity(pattern_name, file_path, line_content)
                    
                    issue = {
                        'file': str(file_path.relative_to(self.project_root)),
                        'line': line_num,
                        'pattern': pattern_name,
                        'content': line_content,
                        'severity': severity,
                        'recommendation': self._get_recommendation(pattern_name)
                    }
                    
                    issues.append(issue)
                    self.summary['total_issues'] += 1
                    
                    if severity == 'critical':
                        self.summary['critical_issues'] += 1
                    elif severity == 'warning':
                        self.summary['warning_issues'] += 1
        
        except Exception as e:
            print(f"Error scanning {file_path}: {e}")
        
        return issues
    
    def _determine_severity(self, pattern_name: str, file_path: Path, line_content: str) -> str:
        """Determine the severity of a timezone issue"""
        file_str = str(file_path)
        
        # Critical issues - affect data storage or trading decisions
        if any(critical_dir in file_str for critical_dir in ['trading/', 'data/', 'ml/']):
            if pattern_name in ['datetime_now_no_tz', 'datetime_fromtimestamp_no_tz']:
                # Check if it's used for timestamps that get stored
                if any(keyword in line_content.lower() for keyword in 
                       ['timestamp', 'created_at', 'updated_at', 'time', 'log']):
                    return 'critical'
        
        # Warning issues - might cause inconsistencies
        if pattern_name in ['datetime_now_no_tz', 'datetime_utcnow']:
            return 'warning'
        
        return 'info'
    
    def _get_recommendation(self, pattern_name: str) -> str:
        """Get recommendation for fixing the issue"""
        recommendations = {
            'datetime_now_no_tz': 'Use datetime.now(timezone.utc) instead',
            'datetime_utcnow': 'Use datetime.now(timezone.utc) instead (utcnow is deprecated)',
            'time_time': 'Consider using datetime.now(timezone.utc).timestamp() for clarity',
            'pd_timestamp_now': 'Use pd.Timestamp.now(tz="UTC") instead',
            'datetime_fromtimestamp_no_tz': 'Add tz=timezone.utc parameter'
        }
        return recommendations.get(pattern_name, 'Review timezone handling')
    
    def check_environment_variables(self) -> Dict[str, Any]:
        """Check timezone-related environment variables"""
        tz_vars = ['TZ', 'PGTZ', 'TIMEZONE']
        env_status = {}
        
        for var in tz_vars:
            value = os.environ.get(var)
            env_status[var] = {
                'set': value is not None,
                'value': value,
                'is_utc': value in ['UTC', 'GMT'] if value else False
            }
        
        return env_status
    
    def generate_report(self) -> str:
        """Generate comprehensive timezone validation report"""
        report = []
        report.append("=" * 80)
        report.append("TIMEZONE VALIDATION REPORT")
        report.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
        report.append("=" * 80)
        
        # PostgreSQL validation
        report.append("\n1. POSTGRESQL TIMEZONE CONFIGURATION")
        report.append("-" * 40)
        pg_result = self.validate_postgresql_timezone()
        
        if pg_result['status'] == 'success':
            if pg_result['is_utc']:
                report.append("✓ PostgreSQL timezone is correctly set to UTC")
            else:
                report.append(f"✗ PostgreSQL timezone issue: {pg_result['issue']}")
        else:
            report.append(f"✗ PostgreSQL validation failed: {pg_result['issue']}")
        
        # Environment variables
        report.append("\n2. ENVIRONMENT VARIABLES")
        report.append("-" * 40)
        env_vars = self.check_environment_variables()
        
        for var, status in env_vars.items():
            if status['set']:
                if status['is_utc']:
                    report.append(f"✓ {var}={status['value']} (UTC)")
                else:
                    report.append(f"⚠ {var}={status['value']} (not UTC)")
            else:
                report.append(f"✓ {var} not set (good)")
        
        # Code analysis
        report.append("\n3. CODE ANALYSIS")
        report.append("-" * 40)
        issues = self.scan_datetime_usage()
        
        report.append(f"Files scanned: {self.summary['total_files_scanned']}")
        report.append(f"Files with issues: {self.summary['files_with_issues']}")
        report.append(f"Total issues found: {self.summary['total_issues']}")
        report.append(f"Critical issues: {self.summary['critical_issues']}")
        report.append(f"Warning issues: {self.summary['warning_issues']}")
        
        # Critical issues
        critical_issues = [issue for issue in issues if issue['severity'] == 'critical']
        if critical_issues:
            report.append("\n4. CRITICAL ISSUES (MUST FIX)")
            report.append("-" * 40)
            for issue in critical_issues:
                report.append(f"✗ {issue['file']}:{issue['line']}")
                report.append(f"  Pattern: {issue['pattern']}")
                report.append(f"  Code: {issue['content']}")
                report.append(f"  Fix: {issue['recommendation']}")
                report.append("")
        
        # Warning issues
        warning_issues = [issue for issue in issues if issue['severity'] == 'warning']
        if warning_issues:
            report.append("\n5. WARNING ISSUES (SHOULD FIX)")
            report.append("-" * 40)
            for issue in warning_issues[:10]:  # Limit to first 10
                report.append(f"⚠ {issue['file']}:{issue['line']}")
                report.append(f"  Pattern: {issue['pattern']}")
                report.append(f"  Code: {issue['content']}")
                report.append(f"  Fix: {issue['recommendation']}")
                report.append("")
            
            if len(warning_issues) > 10:
                report.append(f"... and {len(warning_issues) - 10} more warning issues")
        
        # Summary
        report.append("\n6. SUMMARY")
        report.append("-" * 40)
        
        if self.summary['critical_issues'] == 0:
            report.append("✓ No critical timezone issues found")
        else:
            report.append(f"✗ {self.summary['critical_issues']} critical issues need immediate attention")
        
        if self.summary['warning_issues'] == 0:
            report.append("✓ No warning timezone issues found")
        else:
            report.append(f"⚠ {self.summary['warning_issues']} warning issues should be addressed")
        
        # Recommendations
        report.append("\n7. RECOMMENDATIONS")
        report.append("-" * 40)
        report.append("1. Always use datetime.now(timezone.utc) for UTC timestamps")
        report.append("2. Use datetime.fromtimestamp(ts, tz=timezone.utc) when converting timestamps")
        report.append("3. Avoid datetime.utcnow() as it's deprecated in Python 3.12+")
        report.append("4. Ensure all stored timestamps are in UTC")
        report.append("5. Convert to local timezone only for display purposes")
        
        return "\n".join(report)

def main():
    """Main function to run timezone validation"""
    project_root = os.path.dirname(os.path.abspath(__file__))
    validator = TimezoneValidator(project_root)
    
    print("Running comprehensive timezone validation...")
    report = validator.generate_report()
    
    # Print to console
    print(report)
    
    # Save to file
    report_file = Path(project_root) / "timezone_validation_report.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"\nReport saved to: {report_file}")
    
    # Exit with error code if critical issues found
    if validator.summary['critical_issues'] > 0:
        print(f"\n❌ VALIDATION FAILED: {validator.summary['critical_issues']} critical timezone issues found")
        sys.exit(1)
    else:
        print("\n✅ VALIDATION PASSED: No critical timezone issues found")
        sys.exit(0)

if __name__ == "__main__":
    main()