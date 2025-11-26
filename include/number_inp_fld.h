#include <SFML/Graphics.hpp>
#include <string>
#include <sstream>
#include <type_traits>

template <typename T>
class NumberInputField
{
    static_assert(std::is_arithmetic_v<T>, "T must be a numeric type");

private:
    sf::RectangleShape m_background;
    sf::Text m_text;
    sf::Text m_label;
    sf::Font m_font;
    std::string m_inputString;
    bool m_isSelected;
    T m_value;
    T m_minValue;
    T m_maxValue;
    std::string m_labelText;
    sf::Vector2f m_position;
    sf::Vector2f m_size;

    void updateText()
    {
        m_text.setString(m_inputString + (m_isSelected ? "_" : ""));
    }

    bool isValidInput(const std::string &str) const
    {
        if (str.empty())
            return true;

        std::stringstream ss(str);
        T testValue;
        ss >> testValue;

        // Check if conversion was successful and no characters remain
        return !ss.fail() && ss.eof();
    }

    void updateValue()
    {
        if (m_inputString.empty())
        {
            m_value = T{};
            return;
        }

        std::stringstream ss(m_inputString);
        ss >> m_value;

        // Clamp value to min/max range
        if (m_value < m_minValue)
            m_value = m_minValue;
        if (m_value > m_maxValue)
            m_value = m_maxValue;
    }

public:
    NumberInputField(const sf::Vector2f &position, const sf::Vector2f &size,
                     const sf::Font &font, T minValue, T maxValue,
                     const std::string &labelText = "")
        : m_font(font), m_inputString(), m_isSelected(false), m_value(T{}), m_minValue(minValue), m_maxValue(maxValue), m_labelText(labelText), m_position(position), m_size(size)
    {

        // Setup label (placed to the left of the input field)
        m_label.setFont(font);
        m_label.setCharacterSize(16);
        m_label.setFillColor(sf::Color::White);
        m_label.setString(labelText);

        // Calculate label bounds to position it to the left
        sf::FloatRect labelBounds = m_label.getLocalBounds();
        float labelX = position.x - labelBounds.width - 10; // 10px padding
        float labelY = position.y + (size.y - 20) / 2;
        m_label.setPosition(labelX, labelY);

        // Setup background (input field)
        m_background.setPosition(position);
        m_background.setSize(size);
        m_background.setFillColor(sf::Color::Black);
        m_background.setOutlineThickness(2);
        m_background.setOutlineColor(sf::Color::Green);

        // Setup input text
        m_text.setFont(font);
        m_text.setCharacterSize(16);
        m_text.setFillColor(sf::Color::White);
        m_text.setPosition(position.x + 5, position.y + (size.y - 10) / 2);

        m_font = font;
        updateText();
    }

    void handleEvent(const sf::Event &event)
    {
        if (event.type == sf::Event::MouseButtonPressed)
        {
            sf::Vector2f mousePos(static_cast<float>(event.mouseButton.x),
                                  static_cast<float>(event.mouseButton.y));
            m_isSelected = m_background.getGlobalBounds().contains(mousePos);
            updateText();
        }
        else if (event.type == sf::Event::TextEntered && m_isSelected)
        {
            if (event.text.unicode == '\b')
            { // Backspace
                if (!m_inputString.empty())
                {
                    m_inputString.pop_back();
                }
            }
            else if (event.text.unicode == '\r')
            { // Enter key
                m_isSelected = false;
                updateValue();
            }
            else
            {
                char enteredChar = static_cast<char>(event.text.unicode);
                std::string newString = m_inputString + enteredChar;

                // Allow negative sign only at start and for signed types
                if (enteredChar == '-' && m_inputString.empty() && std::is_signed_v<T>)
                {
                    m_inputString = newString;
                }
                // Allow decimal point only for floating point types
                else if (enteredChar == '.' && std::is_floating_point_v<T>)
                {
                    if (m_inputString.find('.') == std::string::npos)
                    {
                        m_inputString = newString;
                    }
                }
                // Regular digits
                else if (enteredChar >= '0' && enteredChar <= '9')
                {
                    m_inputString = newString;
                }
                // Update numeric value on each key press (so value is immediately available)
                updateValue();
            }
            updateText();
        }
    }
    void draw(sf::RenderWindow &window) const
    {
        window.draw(m_label); // Draw label first (to the left)
        window.draw(m_background);
        window.draw(m_text);
    }

    T getValue() const
    {
        return m_value;
    }

    void setValue(T newValue)
    {
        m_value = newValue;
        if (m_value < m_minValue)
            m_value = m_minValue;
        if (m_value > m_maxValue)
            m_value = m_maxValue;

        m_inputString = std::to_string(m_value);
        updateText();
    }

    // Utility functions for visual customization
    void setBackgroundColor(const sf::Color &color) { m_background.setFillColor(color); }
    void setTextColor(const sf::Color &color) { m_text.setFillColor(color); }
    void setOutlineColor(const sf::Color &color) { m_background.setOutlineColor(color); }
    void setCharacterSize(unsigned int size)
    {
        m_text.setCharacterSize(size);
        m_label.setCharacterSize(size);

        // Recalculate label position after size change
        sf::FloatRect labelBounds = m_label.getLocalBounds();
        float labelX = m_position.x - labelBounds.width - 10;
        float labelY = m_position.y + (m_size.y - size) / 2;
        m_label.setPosition(labelX, labelY);
    }

    // Label-specific customization
    void setLabelColor(const sf::Color &color) { m_label.setFillColor(color); }
    void setLabel(const std::string &newLabel)
    {
        m_labelText = newLabel;
        m_label.setString(newLabel);

        // Recalculate label position after text change
        sf::FloatRect labelBounds = m_label.getLocalBounds();
        float labelX = m_position.x - labelBounds.width - 10;
        m_label.setPosition(labelX, m_label.getPosition().y);
    }
    const std::string &getLabel() const { return m_labelText; }

    // Get bounds for the entire element (label + input field)
    sf::FloatRect getGlobalBounds() const
    {
        sf::FloatRect inputBounds = m_background.getGlobalBounds();
        sf::FloatRect labelBounds = m_label.getGlobalBounds();

        // Return a bounds that encompasses both label and input field
        float left = std::min(inputBounds.left, labelBounds.left);
        float top = std::min(inputBounds.top, labelBounds.top);
        float right = std::max(inputBounds.left + inputBounds.width,
                               labelBounds.left + labelBounds.width);
        float bottom = std::max(inputBounds.top + inputBounds.height,
                                labelBounds.top + labelBounds.height);

        return sf::FloatRect(left, top, right - left, bottom - top);
    }

    // Set a new position for the entire element (both label and input field)
    void setPosition(const sf::Vector2f &newPosition)
    {
        m_position = newPosition;

        // Update input field position
        m_background.setPosition(newPosition);
        m_text.setPosition(newPosition.x + 5, newPosition.y + (m_size.y - m_text.getCharacterSize()) / 2);

        // Update label position
        sf::FloatRect labelBounds = m_label.getLocalBounds();
        float labelX = newPosition.x - labelBounds.width - 10;
        float labelY = newPosition.y + (m_size.y - m_label.getCharacterSize()) / 2;
        m_label.setPosition(labelX, labelY);
    }
};
