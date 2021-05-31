#!/bin/bash

gunicorn SmartCane.wsgi -b 0.0.0.0:8000