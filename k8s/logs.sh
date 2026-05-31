#!/bin/bash

set -e

kubectl logs -l job-name=scaleforge-train -n scaleforge