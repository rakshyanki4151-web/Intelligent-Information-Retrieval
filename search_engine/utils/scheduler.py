import os
from apscheduler.schedulers.background import BackgroundScheduler
from django.conf import settings
from django.core.management import call_command

# Path for the singleton lock file
LOCK_FILE = os.path.join(settings.BASE_DIR, 'crawler.lock')

def scheduled_crawl_job():
    """Industrial Singleton Execution: Monday 11:00 AM"""
    print("\n[SCHEDULED JOB TRIGGERED] Checking for active crawls...")
    
    # 1. Singleton Execution (Race Condition Prevention)
    if os.path.exists(LOCK_FILE):
        print("CRITICAL: Crawler lock exists. Skipping job to prevent overlap.")
        return

    try:
        # Create the lock
        with open(LOCK_FILE, 'w') as f:
            f.write(str(os.getpid()))
        
        print("[LOCK ACQUIRED] Starting Scientific Crawl Process...")
        
        # 4. Management Command Wrapper
        # This calls Command.handle in management/commands/run_crawl.py
        call_command('run_crawl')
        
    except Exception as e:
        print(f"ERR: Scheduler execution failure: {str(e)}")
    finally:
        # 1. Ensure lock is always deleted even on failure
        if os.path.exists(LOCK_FILE):
            os.remove(LOCK_FILE)
            print("[LOCK RELEASED] Scheduler ready for next cycle.")


scheduler = None
scheduler_started = False

def start_scheduler():
    """2. Django Lifecycle Integration: Start the background scheduler"""
    global scheduler, scheduler_started
    
    if scheduler_started:
        return
    
    # APScheduler Background Instance
    scheduler = BackgroundScheduler()
    
    # 2. Schedule for Monday at 11:00 AM
    scheduler.add_job(
        scheduled_crawl_job,
        'cron',
        day_of_week='mon',
        hour=11,
        minute=0,
        id='weekly_scientific_crawl'
    )
    
    scheduler.start()
    scheduler_started = True
    print("SCHEDULER INITIALIZED: Targets every Monday at 11:00 AM")
