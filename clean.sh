#!/bin/bash
rm -rf log/ install/ build/
colcon build --symlink-install
