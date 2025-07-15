#pragma once

#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <stdexcept>

class text_logger
{
public:
    explicit text_logger(const std::string &logFilePath = "")
        : m_log_file_path(logFilePath) {}

    ~text_logger()
    {
        flush_current_line();
    }

    void set_log_file(const std::string &path)
    {
        m_log_file_path = path;
    }

    template <typename T>
    text_logger &operator<<(const T &data)
    {
        m_current_line << data;
        return *this;
    }

    text_logger &operator<<(std::ostream &(*manip)(std::ostream &))
    {
        if (manip == static_cast<std::ostream &(*)(std::ostream &)>(std::endl))
        {
            flush_current_line();
        }
        else
        {
            m_current_line << manip;
        }
        return *this;
    }

    void dump_to_file(bool clearAfter = false, bool append = true)
    {
        if (m_log_file_path.empty())
        {
            throw std::runtime_error("Log file path not set");
        }
        flush_current_line();
        std::ofstream file;
        if (append)
        {
            file.open(m_log_file_path, std::ios_base::app);
        }
        else
        {
            file.open(m_log_file_path);
        }
        if (!file.is_open())
        {
            throw std::runtime_error("Failed to open log file: " + m_log_file_path);
        }
        for (const auto &line : m_buffer)
        {
            file << line << '\n';
        }
        file.close();
        if (clearAfter)
        {
            clear_buffer();
        }
    }

    void dump_to_console(bool clearAfter = false)
    {
        flush_current_line();
        for (const auto &line : m_buffer)
        {
            std::cout << line << '\n';
        }
        if (clearAfter)
        {
            clear_buffer();
        }
    }

    void clear_buffer()
    {
        m_buffer.clear();
    }

    void clear_current_line()
    {
        m_current_line.str("");
        m_current_line.clear();
    }

    void clear_all()
    {
        clear_buffer();
        clear_current_line();
    }

private:
    void flush_current_line()
    {
        std::string line = m_current_line.str();
        if (!line.empty())
        {
            m_buffer.push_back(line);
            clear_current_line();
        }
    }

    std::ostringstream m_current_line;
    std::vector<std::string> m_buffer;
    std::string m_log_file_path;
};
