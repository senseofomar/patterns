import os
from .highlight import highlight_sentence_with_colors, CHAPTERS_FOLDER
from .open_in_pycharm import open_in_pycharm, compute_match_file_line

def interactive_navigation(matches, keywords, kw_color_map, case_sensitive=False):
    if not matches:
        print("No matches found.")
        return

    print(f"\nFound {len(matches)} matches. Showing previews:\n")
    for i,m in enumerate(matches,1):
        p = highlight_sentence_with_colors(m.snippet,[m],keywords,kw_color_map,case_sensitive)
        print(f"{i:04d}. {m.file} - {p}")

    idx = 0
    while True:
        m = matches[idx]
        full = highlight_sentence_with_colors(m.sentence,[m],keywords,kw_color_map,case_sensitive)
        print("\n"+"="*80)
        print(f"Match {idx+1}/{len(matches)} — File: {m.file} — Keyword: {m.keyword}")
        print(full.strip()); print("- Snippet:\n"+m.snippet)
        print("="*80)

        cmd = input("\n[n]ext [p]rev [o]pen [f]ilter [q]uit: ").strip().lower()

        if cmd in ("q","quit","exit"): break
        if cmd in ("","n"): idx=(idx+1)%len(matches); continue
        if cmd=="p": idx=(idx-1)%len(matches); continue

        if cmd=="f":
            s=input("Filter word: ").lower()
            flt=[x for x in matches if s in x.sentence.lower()]
            if flt: matches,idx=flt,0
            else: print("⚠️ No results after filter.")
            continue

        if cmd=="o":
            path=os.path.join(CHAPTERS_FOLDER,m.file)
            line,col=compute_match_file_line(path,m.sentence,m.start)
            open_in_pycharm(path,line=line,column=col)
            continue

        try:
            n=int(cmd)
            if 1<=n<=len(matches): idx=n-1
            else: print("Out of range.")
        except:
            print("Unknown command.")
