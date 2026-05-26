CREATE DATABASE IF NOT EXISTS music_chart
    DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE music_chart;

CREATE TABLE IF NOT EXISTS chart_data (
    id INT AUTO_INCREMENT PRIMARY KEY,
    `rank` INT NOT NULL,
    title VARCHAR(500) NOT NULL,
    artist VARCHAR(300) NOT NULL,
    album VARCHAR(500) DEFAULT '',
    likes INT DEFAULT 0,
    song_id VARCHAR(50) DEFAULT '',
    source VARCHAR(20) NOT NULL,
    crawled_at DATETIME NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_source (source),
    INDEX idx_rank (source, `rank`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS weekly_rank (
    id INT AUTO_INCREMENT PRIMARY KEY,
    week_offset INT NOT NULL,
    week_label VARCHAR(100),
    `rank` INT NOT NULL,
    title VARCHAR(500) NOT NULL,
    artist VARCHAR(300) NOT NULL,
    likes INT DEFAULT 0,
    crawled_at DATETIME NOT NULL,
    INDEX idx_title (title(100)),
    INDEX idx_week (week_offset)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS youtube_stats (
    id INT AUTO_INCREMENT PRIMARY KEY,
    `rank` INT NOT NULL,
    title VARCHAR(500) NOT NULL,
    artist VARCHAR(300) NOT NULL,
    video_title VARCHAR(500),
    video_id VARCHAR(50),
    view_count BIGINT DEFAULT 0,
    like_count BIGINT DEFAULT 0,
    comment_count BIGINT DEFAULT 0,
    comment1 TEXT,
    comment2 TEXT,
    comment3 TEXT,
    comment4 TEXT,
    crawled_at DATETIME NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
