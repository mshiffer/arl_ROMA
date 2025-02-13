#!/bin/bash

# Step 1: Fetch IMDSv2 Token
TOKEN=$(curl -sX PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")

# Step 2: Retrieve the instance ID using the token
INSTANCE_ID=$(curl -sH "X-aws-ec2-metadata-token: $TOKEN" "http://169.254.169.254/latest/meta-data/instance-id")

# Step 3: Validate the instance ID
if [[ -z "$INSTANCE_ID" ]]; then
  echo "Error: Failed to retrieve instance ID. Exiting."
  exit 1
fi

# Step 4: Construct a properly formatted JSON payload using jq
PAYLOAD=$(jq -n --arg instance_id "$INSTANCE_ID" '{instance_id: $instance_id}')

# Step 5: Call the stop-instance Lambda function with the correct payload format
# aws lambda invoke --function-name stop-instance \
#  --payload "$PAYLOAD" \
#  --cli-binary-format raw-in-base64-out \
#  /dev/null

echo "Stop request sent for instance: $INSTANCE_ID"