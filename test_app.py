import unittest
from streamlit.testing.v1 import AppTest

class TestStreamlitDashboard(unittest.TestCase):
    
    def test_initial_load(self):
        """Test that the dashboard loads the default page without any exceptions."""
        at = AppTest.from_file("app.py")
        at.run()
        
        # Assert no exceptions occurred during run
        self.assertFalse(at.exception, f"App threw exception: {at.exception}")
        
        # Verify main header title
        header_found = False
        for markdown_elem in at.markdown:
            if "Credit Card Fraud Detection Dashboard" in str(markdown_elem.value):
                header_found = True
                break
        self.assertTrue(header_found, "Main header title not found on page.")
        
        # Verify navigation options
        nav_radio = at.sidebar.radio[0]
        self.assertEqual(nav_radio.value, "🏠 Dashboard Overview")

    def test_navigation_to_other_pages(self):
        """Test navigation to all pages to make sure they do not crash."""
        at = AppTest.from_file("app.py")
        at.run()
        
        nav_radio = at.sidebar.radio[0]
        
        # Navigation to Performance benchmarks page
        nav_radio.set_value("📊 Model Evaluation & Benchmarks").run()
        self.assertFalse(at.exception, "Crashed when navigating to Benchmarks")
        
        # Navigation to BFA Convergence page
        nav_radio.set_value("🧬 BFA Convergence & Features").run()
        self.assertFalse(at.exception, "Crashed when navigating to Convergence")
        
        # Navigation to Academic Deliverables page
        nav_radio.set_value("📂 Academic Deliverables").run()
        self.assertFalse(at.exception, "Crashed when navigating to Academic Deliverables")

    def test_prediction_playground_genuine_preset(self):
        """Test the interactive prediction form using the Genuine Transaction preset."""
        at = AppTest.from_file("app.py")
        at.run()
        
        # Navigate to Prediction Playground
        nav_radio = at.sidebar.radio[0]
        nav_radio.set_value("🔮 Prediction Playground").run()
        self.assertFalse(at.exception)
        
        # Select Genuine Preset
        preset_select = at.selectbox[0]
        preset_select.set_value("Genuine Transaction Preset").run()
        self.assertFalse(at.exception)
        
        # Check that the number inputs are populated correctly
        # Time is selected_features[0] -> input 0
        time_input = at.number_input[0]
        self.assertEqual(time_input.value, 80000.0)
        
        # Amount is selected_features[-1] -> input -1
        amount_input = at.number_input[-1]
        self.assertEqual(amount_input.value, 15.0)
        
        # Click Predict Form submit button
        # The form submit button is the first button on the page
        predict_button = at.button[0]
        predict_button.click().run()
        self.assertFalse(at.exception)
        
        # Verify that GENUINE TRANSACTION card is shown
        result_found = False
        for markdown_elem in at.markdown:
            if "GENUINE TRANSACTION" in str(markdown_elem.value):
                result_found = True
                break
        self.assertTrue(result_found, "Genuine transaction card was not found after submitting Genuine preset.")

    def test_prediction_playground_fraud_preset(self):
        """Test the interactive prediction form using the Fraudulent Transaction preset."""
        at = AppTest.from_file("app.py")
        at.run()
        
        # Navigate to Prediction Playground
        nav_radio = at.sidebar.radio[0]
        nav_radio.set_value("🔮 Prediction Playground").run()
        self.assertFalse(at.exception)
        
        # Select Fraudulent Preset
        preset_select = at.selectbox[0]
        preset_select.set_value("Fraudulent Transaction Preset").run()
        self.assertFalse(at.exception)
        
        # Time value should be negative -79804.42
        time_input = at.number_input[0]
        self.assertAlmostEqual(time_input.value, -79804.42, places=2)
        
        # Amount value should be 230.04
        amount_input = at.number_input[-1]
        self.assertAlmostEqual(amount_input.value, 230.04, places=2)
        
        # Click Predict Form submit button
        predict_button = at.button[0]
        predict_button.click().run()
        self.assertFalse(at.exception)
        
        # Verify that FRAUDULENT TRANSACTION DETECTED card is shown
        result_found = False
        for markdown_elem in at.markdown:
            if "FRAUDULENT TRANSACTION DETECTED" in str(markdown_elem.value):
                result_found = True
                break
        self.assertTrue(result_found, "Fraudulent transaction card was not found after submitting Fraud preset.")

if __name__ == '__main__':
    unittest.main()
