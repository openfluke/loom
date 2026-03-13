#!/bin/bash
echo "=========================================================="
echo "       Loom NN Docs: Linux/macOS Requirements Setup       "
echo "=========================================================="
echo ""
echo "This script will install Pandoc using your package manager."
echo ""

# Check if pandoc is already installed
if command -v pandoc &> /dev/null; then
    echo "[INFO] Pandoc is already installed: $(pandoc --version | head -n 1)"
    echo "[SUCCESS] You are ready to run: python3 compile_docs.py"
    exit 0
fi

# Determine package manager and install
if command -v apt &> /dev/null; then
    echo "[INFO] Detected Debian/Ubuntu (apt)."
    echo "Running: sudo apt update && sudo apt install -y pandoc"
    sudo apt update && sudo apt install -y pandoc

elif command -v dnf &> /dev/null; then
    echo "[INFO] Detected Fedora/RHEL (dnf)."
    echo "Running: sudo dnf install -y pandoc"
    sudo dnf install -y pandoc

elif command -v pacman &> /dev/null; then
    echo "[INFO] Detected Arch Linux (pacman)."
    echo "Running: sudo pacman -S --noconfirm pandoc"
    sudo pacman -S --noconfirm pandoc

elif command -v brew &> /dev/null; then
    echo "[INFO] Detected macOS/Homebrew (brew)."
    echo "Running: brew install pandoc"
    brew install pandoc

else
    echo "[ERROR] Unsupported package manager."
    echo "Please install pandoc manually from: https://pandoc.org/installing.html"
    exit 1
fi

echo ""
if command -v pandoc &> /dev/null; then
    echo "=========================================================="
    echo "[SUCCESS] Setup complete! Pandoc installed successfully."
    echo "You can now run: python3 compile_docs.py"
    echo "=========================================================="
else
    echo "[ERROR] Installation failed or pandoc is not in your PATH."
    exit 1
fi
