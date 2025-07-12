#pragma once

#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <stdexcept>

class TextLogger {
public:
    explicit TextLogger(const std::string& logFilePath = "")
        : m_logFilePath(logFilePath) {}

    ~TextLogger() {
        flushCurrentLine();
    }

    void setLogFile(const std::string& path) {
        m_logFilePath = path;
    }

    template<typename T>
    TextLogger& operator<<(const T& data) {
        m_currentLine << data;
        return *this;
    }

    TextLogger& operator<<(std::ostream& (*manip)(std::ostream&)) {
        if (manip == static_cast<std::ostream& (*)(std::ostream&)>(std::endl)) {
            flushCurrentLine();
        } else {
            m_currentLine << manip;
        }
        return *this;
    }

    void dumpToFile(bool clearAfter = false, bool append = true) {
        if (m_logFilePath.empty()) {
            throw std::runtime_error("Log file path not set");
        }
        flushCurrentLine();
        std::ofstream file;
        if (append) {
            file.open(m_logFilePath, std::ios_base::app);
        } else {
            file.open(m_logFilePath);
        }
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open log file: " + m_logFilePath);
        }
        for (const auto& line : m_buffer) {
            file << line << '\n';
        }
        file.close();
        if (clearAfter) {
            clearBuffer();
        }
    }

    void dumpToConsole(bool clearAfter = false) {
        flushCurrentLine();
        for (const auto& line : m_buffer) {
            std::cout << line << '\n';
        }
        if (clearAfter) {
            clearBuffer();
        }
    }

    void clearBuffer() {
        m_buffer.clear();
    }

    void clearCurrentLine() {
        m_currentLine.str("");
        m_currentLine.clear();
    }

    void clearAll() {
        clearBuffer();
        clearCurrentLine();
    }

private:
    void flushCurrentLine() {
        std::string line = m_currentLine.str();
        if (!line.empty()) {
            m_buffer.push_back(line);
            clearCurrentLine();
        }
    }

    std::ostringstream m_currentLine;
    std::vector<std::string> m_buffer;
    std::string m_logFilePath;
};
