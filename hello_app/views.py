from flask import Flask, Response
from flask import render_template
from datetime import datetime
from . import app
from pyinstrument import Profiler
import time
import requests

# Functions to simulate a call stack and CPU usage for the flame graph.
def burn_cpu(duration_ms):
    end_time = time.time() + duration_ms / 10000.0
    while time.time() < end_time:
        pass

def a(duration_ms):
    burn_cpu(duration_ms)
    b(duration_ms)
    check_site("http://127.0.0.1:5000/about")

def b(duration_ms):
    burn_cpu(duration_ms)
    burn_cpu(duration_ms)
    check_site("http://127.0.0.1:5000/contact")


def c(duration_ms):
    burn_cpu(duration_ms)
    burn_cpu(duration_ms)
    burn_cpu(duration_ms)
    a(duration_ms)
    b(duration_ms)
    check_site("http://127.0.0.1:5000/")

def check_site(external_url):
    try:
        r = requests.head(external_url)
        # You can access response headers, status code, etc.
        status_code = r.status_code
        content_type = r.headers.get('Content-Type')
        content_length = r.headers.get('Content-Length')

        Response(
            f"HEAD request to {external_url} successful. Status: {status_code}, Content-Type: {content_type}, Content-Length: {content_length}",
            status=status_code,
            content_type="text/plain"
        )
    except requests.exceptions.RequestException as e:
        return Response(f"Error making HEAD request: {e}", status=500, content_type="text/plain")

@app.route("/")
def home():
    # The simulation is now in the flamegraph view.
    return render_template("home.html")

@app.route("/debug/flamegraph")
def flamegraph():
    profiler = Profiler()
    profiler.start()

    # Simulate some work to generate data for the flame graph.
    total_duration_ms = 200
    c(total_duration_ms * 0.8)
    a(total_duration_ms * 0.4)
    b(total_duration_ms * 0.2)
    check_site("http://127.0.0.1:5000/")

    profiler.stop()
    return profiler.output_html()

@app.route("/about/")
def about():
    return render_template("about.html")

@app.route("/contact/")
def contact():
    return render_template("contact.html")

@app.route("/hello/")
@app.route("/hello/<name>")
def hello_there(name = None):
    return render_template(
        "hello_there.html",
        name=name,
        date=datetime.now()
    )

@app.route("/api/data")
def get_data():
    return app.send_static_file("data.json")
