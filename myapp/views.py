import os
import uuid
import threading
from django.shortcuts import render, redirect
from django.http import StreamingHttpResponse
from django.urls import reverse
from django.conf import settings
from .forms import UploadVideoForm
from .models import Violation
from .services.inference import process_video_to_mjpeg

_sessions = {}  # session_id -> thread info

def dashboard(request):
    latest = Violation.objects.all()[:5]
    stats = {
        "total": Violation.objects.count(),
        "no_helmet": Violation.objects.filter(violation_type="no_helmet").count(),
        "overload": Violation.objects.filter(violation_type="overload").count(),
    }
    return render(request, "dashboard.html", {"latest": latest, "stats": stats})

def upload_video(request):
    if request.method == "POST":
        form = UploadVideoForm(request.POST, request.FILES)
        if form.is_valid():
            video_file = form.cleaned_data["video"]
            session_id = uuid.uuid4().hex
            save_path = os.path.join(settings.MEDIA_ROOT, "uploads", f"{session_id}_{video_file.name}")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "wb+") as dest:
                for chunk in video_file.chunks():
                    dest.write(chunk)
            thread = threading.Thread(target=_run_session, args=(save_path, session_id), daemon=True)
            thread.start()
            _sessions[session_id] = {"path": save_path, "thread": thread}
            return redirect(reverse("dashboard") + f"?session={session_id}")
    else:
        form = UploadVideoForm()
    return render(request, "upload.html", {"form": form})

def _run_session(path, session_id):
    # processing occurs via stream generator
    pass

def stream(request, session_id):
    info = _sessions.get(session_id)
    if not info:
        return StreamingHttpResponse(status=404)
    gen = process_video_to_mjpeg(info["path"], session_id)
    return StreamingHttpResponse(gen, content_type="multipart/x-mixed-replace; boundary=frame")

def violations(request):
    qs = Violation.objects.all()
    return render(request, "violations.html", {"violations": qs})