#!/bin/bash

set -e

kubectl get jobs -n scaleforge
kubectl get pods -n scaleforge
kubectl get pvc -n scaleforge