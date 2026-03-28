# ==========================================
# alert_engine.py
# Immediate Visibility Alert Logic
# ==========================================


class AlertEngine:
    def __init__(self, threshold=0.30):
        """
        threshold → visibility threshold
        """
        self.threshold = threshold

    # ------------------------------------------
    # FUNCTION: Check Visibility
    # ------------------------------------------
    def check(self, visibility):
        """
        Input:
            visibility (float 0–1)

        Output:
            True  → trigger alert immediately
            False → no alert
        """

        return visibility < self.threshold


# ------------------------------------------
# Test Block
# ------------------------------------------
if __name__ == "__main__":

    print("Testing Alert Engine...")

    engine = AlertEngine(threshold=0.30)

    test_visibilities = [0.35, 0.28, 0.26, 0.25, 0.40]

    for v in test_visibilities:
        print(f"Visibility: {v} → Alert:", engine.check(v))