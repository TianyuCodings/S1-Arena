"""Make a copy of the balrog-nle 0.9.0 sdist whose NetHack clock can be pinned with NLE_FIXED_TIME (seconds since the epoch,
read as UTC). Unset or empty: the machine's clock, as before. Only getnow() and getlt() in src/hacklib.c change, which feed
the moon phase, Friday the 13th, night() and midnight(); the package version becomes 0.9.0+clockpin."""
import pathlib, shutil, sys

src, dst = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
shutil.rmtree(dst, ignore_errors=True)
shutil.copytree(src, dst, ignore=shutil.ignore_patterns("build"))  # not the earlier build tree
hack = dst / "src" / "hacklib.c"
text = hack.read_text()
old_now = """time_t
getnow()
{
    time_t datetime = 0;

    (void) time((TIME_type) &datetime);
    return datetime;
}"""
new_now = """time_t
getnow()
{
    time_t datetime = 0;
    const char *fixed = getenv("NLE_FIXED_TIME"); /* pinned clock for replayable games */

    if (fixed && *fixed)
        return (time_t) strtoll(fixed, (char **) 0, 10);
    (void) time((TIME_type) &datetime);
    return datetime;
}"""
old_lt = """STATIC_OVL struct tm *
getlt()
{
    time_t date = getnow();

    return localtime((LOCALTIME_type) &date);
}"""
new_lt = """STATIC_OVL struct tm *
getlt()
{
    time_t date = getnow();
    const char *fixed = getenv("NLE_FIXED_TIME");

    if (fixed && *fixed)
        return gmtime((LOCALTIME_type) &date); /* a pinned clock is read as UTC */
    return localtime((LOCALTIME_type) &date);
}"""
assert old_now in text and old_lt in text
text = text.replace(old_now, new_now).replace(old_lt, new_lt)
if "#include <stdlib.h>" not in text:
    text = text.replace('#include "hack.h" /* for config.h+extern.h */', '#include "hack.h" /* for config.h+extern.h */\n#include <stdlib.h>')
hack.write_text(text)
setup = dst / "setup.py"
s = setup.read_text()
assert 'version="0.9.0",' in s
setup.write_text(s.replace('version="0.9.0",', 'version="0.9.0+clockpin",'))
print("patched", dst)
