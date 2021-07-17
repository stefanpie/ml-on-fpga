blank:

.PHONY: clean
clean:
	rm -f *.v
	rm -f *.gtkw
	rm -f *.vcd
	rm -f *.dot
	rm -rf ./build
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf