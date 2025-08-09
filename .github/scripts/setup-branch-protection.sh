#!/bin/bash

# Setup branch protection rules for Git Flow
# This script should be run by a repository admin

echo "🔒 Setting up branch protection rules for Git Flow..."

# Check if gh CLI is installed
if ! command -v gh &> /dev/null; then
    echo "❌ GitHub CLI (gh) is not installed. Please install it first."
    exit 1
fi

# Check if authenticated
if ! gh auth status &> /dev/null; then
    echo "❌ Not authenticated with GitHub. Run 'gh auth login' first."
    exit 1
fi

REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
echo "Repository: $REPO"

# Function to set branch protection
set_branch_protection() {
    local BRANCH=$1
    local ALLOW_FORCE_PUSH=$2
    local REQUIRED_CHECKS=$3
    
    echo "Setting protection for branch: $BRANCH"
    
    # Create the branch protection rule
    gh api \
        --method PUT \
        -H "Accept: application/vnd.github+json" \
        "/repos/$REPO/branches/$BRANCH/protection" \
        -f "required_status_checks[strict]=true" \
        -f "required_status_checks[contexts][]=$REQUIRED_CHECKS" \
        -f "enforce_admins=false" \
        -f "required_pull_request_reviews[dismiss_stale_reviews]=true" \
        -f "required_pull_request_reviews[require_code_owner_reviews]=false" \
        -f "required_pull_request_reviews[required_approving_review_count]=1" \
        -f "required_pull_request_reviews[require_last_push_approval]=false" \
        -f "restrictions=null" \
        -f "allow_force_pushes=$ALLOW_FORCE_PUSH" \
        -f "allow_deletions=false" \
        -f "block_creations=false" \
        -f "required_conversation_resolution=true" \
        -f "lock_branch=false" \
        -f "allow_fork_syncing=true"
}

# Protect main branch
echo "📍 Configuring 'main' branch..."
set_branch_protection "main" "false" "all-tests-pass"

# Additional rules specific to main
gh api \
    --method PATCH \
    -H "Accept: application/vnd.github+json" \
    "/repos/$REPO/branches/main/protection/required_pull_request_reviews" \
    -f "dismiss_stale_reviews=true" \
    -f "require_code_owner_reviews=false" \
    -f "required_approving_review_count=2" \
    -f "require_last_push_approval=true"

# Protect develop branch
echo "📍 Configuring 'develop' branch..."
set_branch_protection "develop" "false" "all-tests-pass"

echo "✅ Branch protection rules configured successfully!"

# Display the rules
echo ""
echo "📋 Current Protection Rules:"
echo ""
echo "🔸 main branch:"
echo "  - Requires PR with 2 approvals"
echo "  - Requires all status checks to pass"
echo "  - Requires branches to be up to date"
echo "  - Requires conversation resolution"
echo "  - No force pushes allowed"
echo "  - No branch deletion allowed"
echo ""
echo "🔸 develop branch:"
echo "  - Requires PR with 1 approval"
echo "  - Requires all status checks to pass"
echo "  - Requires branches to be up to date"
echo "  - Requires conversation resolution"
echo "  - No force pushes allowed"
echo "  - No branch deletion allowed"
echo ""
echo "🎯 Git Flow is now enforced!"