cmd_fs/ext4/ext4.o := arm-none-linux-gnueabi-ld -EL    -r -o fs/ext4/ext4.o fs/ext4/balloc.o fs/ext4/bitmap.o fs/ext4/dir.o fs/ext4/file.o fs/ext4/fsync.o fs/ext4/ialloc.o fs/ext4/inode.o fs/ext4/ioctl.o fs/ext4/namei.o fs/ext4/super.o fs/ext4/symlink.o fs/ext4/hash.o fs/ext4/resize.o fs/ext4/extents.o fs/ext4/ext4_jbd2.o fs/ext4/migrate.o fs/ext4/mballoc.o fs/ext4/block_validity.o fs/ext4/move_extent.o fs/ext4/xattr.o fs/ext4/xattr_user.o fs/ext4/xattr_trusted.o 
