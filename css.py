jupyter_like_style = """
    table {
      border: none;
      border-collapse: collapse;
      border-spacing: 0;
      color: black;
      font-size: 12px;
      table-layout: fixed;
    }
    thead {
      border-bottom: 1px solid black;
      vertical-align: bottom;
    }
    tr, th, td {
      text-align: right;
      vertical-align: middle;
      padding: 0.5em 0.5em;
      line-height: normal;
      white-space: normal;
      max-width: none;
      border: none;
    }
    th {
      font-weight: bold;
    }
    tbody tr:nth-child(odd) {
      background: #f5f5f5;
    }
    tbody tr:hover {
      background: rgba(66, 165, 245, 0.2);
    }
    .left {
      float: left;
    }
    .clear {
      clear: left
    }
"""