import subprocess


def get_git_info():
    try:
        # Get the current commit hash
        commit_hash = subprocess.run(
            ["git", "rev-parse", "HEAD"], stdout=subprocess.PIPE, text=True, check=True
        ).stdout.strip()

        # Get the remote URL (assumes GitHub; adjust parsing if using a different service)
        remote_url = subprocess.run(
            ["git", "config", "--get", "remote.origin.url"],
            stdout=subprocess.PIPE,
            text=True,
            check=True,
        ).stdout.strip()

        # Clean up the remote URL (handle SSH or HTTPS URLs)
        if remote_url.startswith("git@"):
            # Convert SSH to HTTPS
            remote_url = remote_url.replace(":", "/").replace("git@", "https://")
        if remote_url.endswith(".git"):
            remote_url = remote_url[:-4]

        # Generate the commit URL
        commit_url = f"{remote_url}/commit/{commit_hash}"

        return commit_hash, commit_url
    except subprocess.CalledProcessError as e:
        print("Error retrieving Git information:", e)
        return None, None


# Example usage in logging
if __name__ == "__main__":
    commit_hash, commit_url = get_git_info()

    if commit_hash and commit_url:
        print(f"Current commit hash: {commit_hash}")
        print(f"GitHub commit URL: {commit_url}")
    else:
        print("Could not retrieve Git information.")
