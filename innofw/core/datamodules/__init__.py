"""
This package is part of our framework's CORE, which is meant to give flexible support for data loading from different
libraries and frameworks via common abstract wrapper, currently it has support for:
- pytorch lightning
- pandas

A datamodule is a shareable, reusable class that encapsulates all the steps needed to process data.

A datamodule encapsulates the five steps involved in data processing:
Download / tokenize / process.
Clean and (maybe) save to disk.
Load inside Dataset.
Apply transforms (rotate, tokenize, etc…).
Wrap inside a DataLoader.
"""
