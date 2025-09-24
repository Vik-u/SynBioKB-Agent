from biosyn_kb.extract.clean import extract_clean_text


def test_extract_clean_text_removes_boilerplate():
    html = """
    <html><head><title>T</title></head>
    <body>
      <nav>Home | About | Cookie Notice</nav>
      <article>
        <h1>Isobutanol production</h1>
        <p>We achieved a titer of 100 g/L in a fed-batch fermentation.</p>
      </article>
      <footer>Contact | Privacy</footer>
    </body></html>
    """
    out = extract_clean_text(html)
    assert "titer of 100 g/L" in out.text
    assert "Cookie" not in out.text

