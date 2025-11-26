#pragma once

#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <stdexcept>

/**
 * @brief A thread-unsafe logger class that buffers log lines and supports output to file/console
 *
 * Usage examples:
 * @code
 * text_logger logger("log.txt");
 * logger << "Error: " << 404 << " Not Found" << std::endl;
 * logger.dump_to_file(); // Writes buffered lines to file
 *
 * text_logger console_logger;
 * console_logger << "Warning: Resource low" << std::endl;
 * console_logger.dump_to_console(); // Outputs to std::cout
 * @endcode
 */
class text_logger
{
public:
    /**
     * @brief Construct a new text logger object
     * @param logFilePath Optional path to log file (can be set later with set_log_file())
     */
    explicit text_logger(const std::string &logFilePath = "")
        : m_log_file_path(logFilePath) {}

    /**
     * @brief Destructor - automatically flushes any pending line to buffer
     */
    ~text_logger()
    {
        flush_current_line();
    }

    /**
     * @brief Set the output log file path
     * @param path Filesystem path for log output
     */
    void set_log_file(const std::string &path)
    {
        m_log_file_path = path;
    }

    /**
     * @brief Stream insertion operator for log messages
     * @tparam T Type of data to log
     * @param data Data to append to current line
     * @return Reference to self for chaining
     *
     * @code
     * logger << "Value: " << 42 << " check";
     * @endcode
     */
    template <typename T>
    text_logger &operator<<(const T &data)
    {
        m_current_line << data;
        return *this;
    }

    /**
     * @brief Handle stream manipulators (e.g., std::endl)
     * @param manip Stream manipulator function
     * @return Reference to self for chaining
     *
     * @note Only std::endl triggers line flushing. Other manipulators (e.g., std::hex)
     *       are passed through to the internal string stream.
     */
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

    /**
     * @brief Write all buffered lines to the log file
     * @param clearAfter If true, clears buffer after writing
     * @param append If true, appends to file; otherwise overwrites
     * @throws std::runtime_error if file path not set or file cannot be opened
     */
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

    /**
     * @brief Output all buffered lines to std::cout
     * @param clearAfter If true, clears buffer after output
     */
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

    /**
     * @brief Clear all buffered lines (does not affect current line in progress)
     */
    void clear_buffer()
    {
        m_buffer.clear();
    }

    /**
     * @brief Clear the current line being built
     */
    void clear_current_line()
    {
        m_current_line.str("");
        m_current_line.clear();
    }

    /**
     * @brief Clear both buffer and current line
     */
    void clear_all()
    {
        clear_buffer();
        clear_current_line();
    }

private:
    /**
     * @brief Moves the current line from m_current_line to m_buffer
     */
    void flush_current_line()
    {
        std::string line = m_current_line.str();
        if (!line.empty())
        {
            m_buffer.push_back(line);
            clear_current_line();
        }
    }

    std::ostringstream m_current_line; ///< Buffer for current line being built
    std::vector<std::string> m_buffer; ///< Storage for completed lines
    std::string m_log_file_path;       ///< Path to output log file
};