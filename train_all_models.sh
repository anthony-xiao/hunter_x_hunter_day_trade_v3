#!/bin/bash

# Script to train models for all symbols in the trading universe

echo "Starting bulk model training for all symbols..."
echo "Timestamp: $(date)"

# Get the list of symbols from the API
symbols=$(curl -s -X GET "http://localhost:8000/data/universe" | jq -r '.symbols[]')

if [ -z "$symbols" ]; then
    echo "Error: Could not retrieve symbols from API"
    exit 1
fi

echo "Found symbols: $(echo $symbols | tr '\n' ' ')"
echo "Total symbols: $(echo "$symbols" | wc -l)"
echo ""

# Counter for tracking progress
count=0
total=$(echo "$symbols" | wc -l)

# Train models for each symbol
for symbol in $symbols; do
    count=$((count + 1))
    echo "[$count/$total] Training models for $symbol..."
    
    # Make the training request
    response=$(curl -s -X POST "http://localhost:8000/models/train/$symbol")
    
    # Check if the request was successful
    if echo "$response" | jq -e '.message' > /dev/null 2>&1; then
        echo "  ✓ Training started for $symbol"
        echo "  Response: $(echo $response | jq -r '.message')"
    else
        echo "  ✗ Failed to start training for $symbol"
        echo "  Response: $response"
    fi
    
    echo ""
    
    # Add a small delay to avoid overwhelming the server
    sleep 2
done

echo "Bulk training requests completed!"
echo "Note: Training runs in the background. Use the performance endpoints to check progress."
echo ""
echo "To check training progress for a specific symbol:"
echo "curl -X GET 'http://localhost:8000/models/performance/SYMBOL_NAME'"
echo ""
echo "To check data status:"
echo "curl -X GET 'http://localhost:8000/data/status'"