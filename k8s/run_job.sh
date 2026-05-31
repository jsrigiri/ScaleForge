#!/bin/bash

set -e

kubectl delete job scaleforge-train -n scaleforge --ignore-not-found=true
kubectl apply -f k8s/training-job.yaml

kubectl get jobs -n scaleforge
kubectl get pods -n scaleforge