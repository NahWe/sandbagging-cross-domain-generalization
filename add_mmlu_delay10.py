path = "src/data/mmlu.py"
src = open(path).read()
src = src.replace("time.sleep(3)", "time.sleep(10)")
open(path, "w").write(src)
print("delay bumped to 10s")
