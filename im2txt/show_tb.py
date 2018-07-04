import traceback

# Shows a traceback
def show_tb():
  # Raise an error to generate a traceback
  try:
    raise TypeError("Intentional error")
  except Exception:
    print(traceback.format_exc())