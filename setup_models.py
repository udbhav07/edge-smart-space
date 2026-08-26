import openwakeword


if __name__ == "__main__":
    print("Downloading OpenWakeWord models...")
    openwakeword.utils.download_models()
    print("OpenWakeWord models are ready.")